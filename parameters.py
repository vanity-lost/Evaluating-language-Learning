TRAIN_DIR = {
    'current': './dataset/current/train.csv',
    'old': './dataset/old/train.csv',
    'old2': './dataset/old-2/train.csv',
    'hewlett': './dataset/hewlett/train.csv'
}
TEST_DIR = 'test.csv'
SUBMISSION_DIR = 'submission.csv'

# Embedding model
DEBERTA = 'microsoft/deberta-v3-base'
DEBERTA_DIR = './deberta'

# Embedding training
MAX_LENGTH = 512
BATCH_SIZE = 4
TRAIN_EMBEDDING = 'train_embeddings.npy'
TEST_EMBEDDING = 'test_embeddings.npy'
