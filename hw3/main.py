from importlib import import_module

preprocess_features = import_module('preprocess-features')
preprocess_vocab = import_module('preprocess-vocab')

import training_procedure
import evaluation

if __name__ == '__main__':
    preprocess_features.main()
    preprocess_vocab.main()
    training_procedure.main()
    evaluation.evaluate_hw3()
