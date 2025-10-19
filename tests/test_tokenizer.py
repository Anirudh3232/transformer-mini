from transformer_tiny.tokenizer import BOS_ID, EOS_ID, PAD_ID, ByteTokenizer


def test_roundtrip():
    tok = ByteTokenizer()
    s = "Hello_2025"
    ids = tok.encode(s)
    # Contains BOS/EOS
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    # Round-trip ignores BOS/EOS
    dec = tok.decode(ids)
    assert dec == s
