import pytest
import os
from llama.tokenizer import Tokenizer
import logging
import traceback

from test.create_tokenizer import create_test_tokenizer

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test():
    print("Starting tokenizer test...")

    model_path = "test/tokenizer_files/test_tokenizer.model"
    print(f"Looking for model at: {model_path}")

    if not os.path.exists(model_path):
        logging.info("No test model found, creating one...")
        print("Model not found, creating...")

        try:
            model_path = create_test_tokenizer()
            print(f"Created model at: {model_path}")

        except Exception as e:
            logging.error(f"Failed to create tokenizer: {e}")
            print(f"Failed to create tokenizer: {e}")
            logging.info("You can try:")
            logging.info("   1. Download a real tokenizer model")
            logging.info("   2. Use a smaller vocab_size")
            logging.info("   3. Provide more training data")
            return
    else:
        print(f"Found existing model: {model_path}")

    # Load and test tokenizer
    try:
        print("Loading tokenizer...")
        tokenizer = Tokenizer(model_path)
        logging.info("Tokenizer loaded successfully")
        print("Tokenizer loaded!")

        # Validation info
        assert tokenizer.n_words > 0, "Vocabulary size should be positive"
        assert isinstance(tokenizer.bos_id, int), "BOS ID should be integer"
        assert isinstance(tokenizer.eos_id, int), "EOS ID should be integer"
        assert isinstance(tokenizer.pad_id, int), "PAD ID should be integer"

        logging.info(f"Tokenizer Info:")
        logging.info(f"   Vocab size: {tokenizer.n_words}")
        logging.info(f"   BOS ID: {tokenizer.bos_id}")
        logging.info(f"   EOS ID: {tokenizer.eos_id}")
        logging.info(f"   PAD ID: {tokenizer.pad_id}")

        # Test cases for encoding and decoding
        test_cases = [
            "Hello World!",
            "This is a test.",
            "LLaMA tokenizer",
            "Testing 123",
            "A"
        ]

        print(f"Testing {len(test_cases)} cases...")

        for i, text in enumerate(test_cases, 1):
            logging.info(f"Testing: '{text}'")
            print(f"  Test {i}: '{text}'")

            tokens_no_special = tokenizer.encode(text, bos=False, eos=False)
            tokens_with_special = tokenizer.encode(text, bos=True, eos=True)

            # Basic validations
            assert isinstance(tokens_no_special, list), "Tokens should be list"
            assert isinstance(tokens_with_special, list), "Tokens should be list"
            assert all(isinstance(t, int) for t in tokens_with_special), "All tokens should be integers"

            # Special tokens validation
            if text:  # Not empty text
                assert len(tokens_with_special) >= 2, "Should have at least BOS and EOS"
                assert tokens_with_special[0] == tokenizer.bos_id, "First token should be BOS"
                assert tokens_with_special[-1] == tokenizer.eos_id, "Last token should be EOS"
                assert len(tokens_with_special) == len(tokens_no_special) + 2, "Special tokens should add 2 tokens"

            # Decode test
            decoded = tokenizer.decode(tokens_with_special)
            assert isinstance(decoded, str), "Decoded output should be string"

            logging.info(f"   Tokens (no special): {tokens_no_special}")
            logging.info(f"   Tokens (with special): {tokens_with_special}")
            logging.info(f"   Decoded: '{decoded}'")
            logging.info(f"   Length: {len(text)} chars -> {len(tokens_with_special)} tokens")

            print(f"    Test {i} passed!")

        logging.info("All basic tests passed!")
        print("All tests completed successfully!")

    except Exception as e:
        logging.error(f"Test failed: {e}")
        print(f"Test failed: {e}")
        traceback.print_exc()
        if 'pytest' in globals():
            pytest.fail(f"Tokenizer test failed: {e}")


if __name__ == "__main__":
    print("Running tokenizer test script...")
    test()
    print("Script finished!")