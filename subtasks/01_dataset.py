import os, sys
sys.path.append(os.getcwd())


def main():
    from datasets.hograspnet_toy import HOGraspNetToyDataset
    from datasets.augmentation import RandomYawRotation, RandomJitter
    from utils.data_utils import ContrastiveTransformations, custom_collate_fn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    configs = {
        'data': {
            'root': "/workspace/lab_intern/KDW/GraspRep/data/densehograspnet_pointcloud",
            'setup': 'full',
            'test_ratio': 0.15,
            'split_trials': 100,
        },
    }

    aug_transforms = T.Compose([
        RandomYawRotation(),
        RandomJitter()
    ])


    batch_size = 16

    ds = HOGraspNetToyDataset(root=configs['data']['root'],
                              split='train',
                              setup=configs['data']['setup'],
                              test_ratio=configs['data']['test_ratio'],
                              split_trials=configs['data']['split_trials'],
                              transform=aug_transforms)
    
    # ContrastiveTransformations 안 쓰면 [16, 3, 4096] 형태로 나옴.
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    batch = next(iter(dl))

    print(batch[0].shape)

    transform = ContrastiveTransformations(aug_transforms, n_views=2)

    ds = HOGraspNetToyDataset(root=configs['data']['root'],
                              split='train',
                              setup=configs['data']['setup'],
                              test_ratio=configs['data']['test_ratio'],
                              split_trials=configs['data']['split_trials'],
                              transform=transform)
    
    dl = DataLoader(ds, batch_size=batch_size // 2, shuffle=True, collate_fn=custom_collate_fn)

    batch = next(iter(dl))

    print(batch[0].shape) # (16, 2, 3, 4096)

if __name__ == '__main__':
    main()