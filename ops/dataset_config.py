import os
ROOT_DATASET = 'your_data_path/dataset/'

def return_kinetics(modality):
    filename_categories = 'kinect400/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + "kinect400"
        filename_imglist_train = "kinect400/train_videofolder.txt"
        filename_imglist_val = "kinect400/val_videofolder.txt"
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_something(modality):
    filename_categories = 'something/category.txt'
    if modality == 'RGB' or modality== 'RGBDiff':
        root_data = ROOT_DATASET + "something/20bn-something-something-v1" 
        filename_imglist_train = "something/train_videofolder.txt"
        filename_imglist_val = "something/val_videofolder.txt"
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/train_videofolder_flow.txt'
        filename_imglist_val = 'something/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_somethingv2(modality):
    filename_categories = 174 
    if modality == 'RGB':
        root_data = ROOT_DATASET + "somethingv2/20bn-something-something-v2-frames"
        filename_imglist_train = "somethingv2/train_videofolder.txt"
        filename_imglist_val = "somethingv2/val_videofolder.txt"
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'somethingv2/20bn-something-something-v2-flow'
        filename_imglist_train = 'somethingv2/train_videofolder_flow.txt'
        filename_imglist_val = 'somethingv2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(modality):
    filename_categories = 'Jester/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Jester/20bn-jester-v1'
        filename_imglist_train = 'Jester/train_videofolder.txt'
        filename_imglist_val = 'Jester/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    dict_single = {'kinetics': return_kinetics, 'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
