# coding: utf-8
def __scale_each_test(trainset, testset):
    """Scale each case against all train instances individually (no interference between test cases"""
    scaled_test = []
    for tc in testset:
        scaled = preprocessing.scale(trainset + [tc])
        scaled_test.append(scaled[-1])
    return np.array(scaled_test)

def __scale_test_train_together(trainset, testset):
    all_scaled = preprocessing.scale(trainset + testset)
    return (all_scaled[:len(trainset)], all_scaled[len(trainset):])
    
