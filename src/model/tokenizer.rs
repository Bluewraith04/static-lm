use std::collections::HashMap;

use crate::memory::loader::{GgufError, GgufFile};

/// Embedded GGUF BPE / SentencePiece tokenizer loaded directly from `GgufFile::metadata`.
#[derive(Clone, Debug)]
pub struct GgufTokenizer {
    pub vocab: Vec<String>,
    pub scores: Vec<f32>,
    token_to_id: HashMap<String, usize>,
    merge_ranks: HashMap<(usize, usize), usize>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub unk_token_id: usize,
    uses_sp_space: bool,
}

impl GgufTokenizer {
    /// Constructs a [`GgufTokenizer`] from `tokenizer.ggml.*` metadata inside a [`GgufFile`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let tokens_val = gguf
            .get_str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| GgufError::TensorNotFound("tokenizer.ggml.tokens".to_owned()))?;

        let mut vocab = Vec::with_capacity(tokens_val.len());
        let mut token_to_id = HashMap::with_capacity(tokens_val.len());
        let mut uses_sp_space = false;

        for (idx, &s) in tokens_val.iter().enumerate() {
            if s.contains('▁') {
                uses_sp_space = true;
            }
            vocab.push(s.to_owned());
            token_to_id.entry(s.to_owned()).or_insert(idx);
        }

        let scores = gguf
            .get_f32_array("tokenizer.ggml.scores")
            .unwrap_or_else(|| (0..vocab.len()).map(|i| -(i as f32)).collect());

        let mut merge_ranks = HashMap::new();
        if let Some(merges_arr) = gguf.get_str_array("tokenizer.ggml.merges") {
            for (rank, rule) in merges_arr.into_iter().enumerate() {
                if let Some((left, right)) = rule.split_once(' ') {
                    if let (Some(&l_id), Some(&r_id)) =
                        (token_to_id.get(left), token_to_id.get(right))
                    {
                        merge_ranks.insert((l_id, r_id), rank);
                    }
                }
            }
        }

        let bos_token_id = gguf
            .get_usize("tokenizer.ggml.bos_token_id")
            .unwrap_or(1)
            .min(vocab.len().saturating_sub(1));
        let eos_token_id = gguf
            .get_usize("tokenizer.ggml.eos_token_id")
            .unwrap_or(2)
            .min(vocab.len().saturating_sub(1));
        let unk_token_id = gguf
            .get_usize("tokenizer.ggml.unknown_token_id")
            .unwrap_or(0)
            .min(vocab.len().saturating_sub(1));

        Ok(Self {
            vocab,
            scores,
            token_to_id,
            merge_ranks,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            uses_sp_space,
        })
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encodes `text` into token IDs using character lookup followed by iterative BPE merges.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        let mut out = Vec::new();
        if add_bos && !self.vocab.is_empty() {
            out.push(self.bos_token_id);
        }
        if text.is_empty() {
            return out;
        }

        let normalized = if self.uses_sp_space {
            format!("▁{}", text.replace(' ', "▁"))
        } else {
            text.to_owned()
        };

        let mut symbols = Vec::with_capacity(normalized.len());
        let mut buf = [0u8; 4];
        for ch in normalized.chars() {
            let ch_str = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.token_to_id.get(ch_str) {
                symbols.push(id);
            } else {
                for &b in ch_str.as_bytes() {
                    let hex_tok = format!("<0x{b:02X}>");
                    let id = self
                        .token_to_id
                        .get(&hex_tok)
                        .copied()
                        .unwrap_or(self.unk_token_id);
                    symbols.push(id);
                }
            }
        }

        // Iteratively merge the best adjacent pair according to explicit BPE merge ranks or token scores
        while symbols.len() >= 2 {
            let mut best_idx = None;
            let mut best_rank = usize::MAX;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_merged_id = 0usize;

            for i in 0..(symbols.len() - 1) {
                let l_id = symbols[i];
                let r_id = symbols[i + 1];
                let merged_str = format!("{}{}", self.vocab[l_id], self.vocab[r_id]);
                let Some(&merged_id) = self.token_to_id.get(&merged_str) else {
                    continue;
                };

                if !self.merge_ranks.is_empty() {
                    if let Some(&rank) = self.merge_ranks.get(&(l_id, r_id)) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                            best_merged_id = merged_id;
                        }
                    }
                } else {
                    let score = self.scores.get(merged_id).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(i);
                        best_merged_id = merged_id;
                    }
                }
            }

            if let Some(idx) = best_idx {
                symbols[idx] = best_merged_id;
                symbols.remove(idx + 1);
            } else {
                break;
            }
        }

        out.extend_from_slice(&symbols);
        out
    }

    /// Decodes a single token ID into its string piece.
    pub fn decode_token(&self, token_id: usize) -> String {
        let Some(raw) = self.vocab.get(token_id) else {
            return String::new();
        };
        if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
            if let Ok(byte) = u8::from_str_radix(&raw[3..5], 16) {
                return String::from_utf8_lossy(&[byte]).into_owned();
            }
        }
        raw.replace('▁', " ").replace('Ġ', " ")
    }

    /// Decodes a sequence of token IDs into a UTF-8 string (skipping BOS/EOS control tokens).
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut s = String::new();
        for &t in tokens {
            if t == self.bos_token_id || t == self.eos_token_id {
                continue;
            }
            s.push_str(&self.decode_token(t));
        }
        if self.uses_sp_space && s.starts_with(' ') {
            s.remove(0);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::loader::GgufWriter;

    #[test]
    fn test_gguf_tokenizer_bpe_encode_decode_roundtrip() {
        let path = std::env::temp_dir().join("static_lm_tokenizer_test.gguf");
        let vocab = [
            "<unk>", "<s>", "</s>", "▁", "s", "t", "a", "i", "c", "l", "m", "▁s", "▁st",
            "▁sta", "▁stat", "▁stati", "▁static", "▁l", "▁lm",
        ];
        let scores: Vec<f32> = (0..vocab.len()).map(|i| i as f32).collect();
        let merges = [
            "▁ s",
            "▁s t",
            "▁st a",
            "▁sta t",
            "▁stat i",
            "▁stati c",
            "▁ l",
            "▁l m",
        ];

        let mut writer = GgufWriter::new();
        writer.add_str_array("tokenizer.ggml.tokens", &vocab);
        writer.add_f32_array("tokenizer.ggml.scores", &scores);
        writer.add_str_array("tokenizer.ggml.merges", &merges);
        writer.add_u32("tokenizer.ggml.bos_token_id", 1);
        writer.add_u32("tokenizer.ggml.eos_token_id", 2);
        writer.add_u32("tokenizer.ggml.unknown_token_id", 0);
        writer.write_to_file(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let tokenizer = GgufTokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tokenizer.vocab_size(), vocab.len());

        let ids_with_bos = tokenizer.encode("static lm", true);
        // Expected: BOS (1), "▁static" (16), "▁lm" (18)
        assert_eq!(ids_with_bos, vec![1, 16, 18]);

        let decoded = tokenizer.decode(&ids_with_bos);
        assert_eq!(decoded, "static lm");
    }
}
