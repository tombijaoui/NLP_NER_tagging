import preprocessing as pp
from FFNmodel import train_FFN, test_FFN, predict_FFN
from torch.optim import Adam
from torch import nn
from gensim import downloader


class CompFFN(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_classes):
        super(CompFFN, self).__init__()
        self.first_layer = nn.Linear(input_dimension, hidden_dimension)
        self.second_layer = nn.Linear(hidden_dimension, hidden_dimension)
        self.third_layer = nn.Linear(hidden_dimension, num_classes)

        self.first_activation = nn.LeakyReLU()
        self.second_activation = nn.LeakyReLU()
        self.third_activation = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, labels):
        X = self.first_layer(X)
        X = self.first_activation(X)

        X = self.second_layer(X)
        X = self.second_activation(X)

        X = self.third_layer(X)
        y_pred = self.third_activation(X)

        if labels is None:
            return y_pred, None

        loss = self.loss(labels, y_pred)
        return y_pred, loss


def main():
    GLOVE_PATH = 'glove-twitter-200'
    train_path = 'train.tagged'
    val_path = 'dev.tagged'
    test_path = 'test.untagged'

    # Downloading glove
    print("Downloading GloVe...")
    glove_model = downloader.load(GLOVE_PATH)
    print("GloVe downloaded successfully !\n")

    # Preprocessing
    print("Preprocessing...")
    X_train, y_train = pp.get_words_and_tags("train.tagged", True)

    X_validation, y_validation = pp.get_words_and_tags("dev.tagged", True)

    X_test = pp.get_words_and_tags("test.untagged", False)

    X_train = pp.words2glove(X_train, y_train, glove_model)
    X_validation = pp.words2glove(X_validation, y_validation, glove_model)
    X_test = pp.words2glove(X_test, None, glove_model)

    print("Preprocessing is done !\n")

    batch_size = 16
    num_epochs = 35

    model_2 = CompFFN(input_dimension=len(X_train[0]), hidden_dimension=100, num_classes=2)
    optimizer = Adam(params=model_2.parameters())

    train_FFN(model_2, X_train, y_train, optimizer, batch_size, num_epochs)
    print("test on train:\n")
    test_FFN(model_2, X_train, y_train, batch_size)
    print("test on validation:\n")
    test_FFN(model_2, X_validation, y_validation, batch_size)

    # Test part on the test.untagged file and creation of the predicted_file
    predict_FFN(model_2, X_test, optimizer, batch_size, num_epochs, 'comp_342791324_931214522.tagged')


if __name__ == '__main__':
    main()
