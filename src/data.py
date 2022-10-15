import sys
import omniglot


"""
General function that selects and initializes the particular dataset to use for
few-shot classification. Additional dataset support should be added here.
"""


def get_data(dataset, mode='train'):
    if dataset == 'Omniglot':
        return omniglot.OmniglotData(path='../data/omniglot.npy',
                                     train_size=1100,
                                     validation_size=100,
                                     augment_data=True,
                                     seed=111)
    else:
        sys.exit("Unsupported dataset type (%s)." % dataset)
