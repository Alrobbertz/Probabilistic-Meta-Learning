import sys
import omniglot
import nab


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
    if dataset == 'NAB':
        return nab.NumentaData(path='../data/nab.npy',
                                     train_size=200,
                                     validation_size=75,
                                     augment_data=False,
                                     seed=111)
    else:
        sys.exit("Unsupported dataset type (%s)." % dataset)
