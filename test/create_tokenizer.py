import sentencepiece as smp
import tempfile
import os
import logging

def create_test_tokenizer():
    sample_text = [
        "Hello world", "This is a test", "LLaMA tokenizer test",
        "Machine learning is awesome", "Natural language processing",
        "PyTorch implementation", "Transformer architecture",
        "Attention mechanism", "Neural networks rule", "Deep learning rocks",
        "Artificial intelligence", "Computer vision", "Speech recognition",
        "Language models", "GPT ChatGPT", "BERT RoBERTa", "Text generation",
        "Sequence to sequence", "Encoder decoder", "Self attention",
        "Multi head attention", "Feed forward network", "Layer normalization",
        "Dropout regularization", "Adam optimizer", "Learning rate scheduling",
        "Gradient descent", "Backpropagation", "Neural network training",
        "Data preprocessing", "Tokenization process", "Vocabulary building",
        "Subword segmentation", "Byte pair encoding", "SentencePiece model",
        "Text classification", "Named entity recognition", "Question answering",
        "Text summarization", "Machine translation", "Language understanding",
        "The quick brown fox", "jumps over the lazy dog", "Pack my box",
        "with five dozen liquor jugs", "How razorback jumping frogs",
        "can level six piqued gymnasts", "Crazy Fredrick bought",
        "many very exquisite opal jewels", "We promptly judged antique",
        "ivory buckles for the next prize", "A mad boxer shot",
        "a quick gloved jab", "Few quips galvanized the mock",
        "jury box", "Quick brown dogs jump over",
        "the lazy fox", "The five boxing wizards",
        "jump quickly", "Bright vixens jump",
        "Pack my red box", "How quickly daft jumping",
        "zebras vex", "Quick zephyrs blow",
        "vexing daft Jim", "Waltz nymph for quick",
        "jigs vex bud", "Sphinx of black quartz",
        "judge my vow", "The quick onyx goblin",
        "jumps over the lazy dwarf", "Jackdaws love my big",
        "sphinx of quartz", "Pack my box with",
        "five dozen liquor jugs", "We promptly judged",
        "antique ivory buckles", "Crazy Fredrick bought many",
        "very exquisite opal jewels", "A mad boxer shot a",
        "quick gloved jab to", "the jaw of his",
        "dizzy opponent", "The job requires extra",
        "pluck and zeal from", "every young wage earner",
        "Quick brown fox jumps", "over the lazy dog",
        "Programming languages include Python", "Java JavaScript TypeScript",
        "C++ C# Rust Go", "HTML CSS SQL", "React Vue Angular",
        "Node.js Express Django", "Flask FastAPI Spring", "Docker Kubernetes",
        "Git GitHub GitLab", "VS Code IntelliJ", "Machine learning frameworks",
        "TensorFlow PyTorch Keras", "Scikit-learn Pandas NumPy", "Matplotlib Seaborn Plotly",
        "Jupyter Notebook Colab", "AWS Azure GCP", "Linux Windows MacOS",
        "Bash Shell PowerShell", "Database MySQL PostgreSQL", "MongoDB Redis Elasticsearch",
        "API REST GraphQL", "JSON XML YAML", "HTTP HTTPS WebSocket",
        "Authentication OAuth JWT", "Testing unit integration", "CI/CD pipeline deployment"
    ]

    sample_texts = []
    for text in sample_text:
        sample_texts.extend([text] * 10)

    # temp file save
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
        temp_file = f.name

    try:
        logging.info("Creating test tokenizer...")

        smp.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix="test_tokenizer",
            vocab_size=1000,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            character_coverage=1.0,
            split_digits=True
        )

        logging.info("test_tokenizer.model created!")
        return "test_tokenizer.model"

    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    create_test_tokenizer()