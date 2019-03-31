import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as datasets
from models.utils_cifar import train, test, std, mean, get_hms

def get_loader_cifar(loader, loader_type, cifar_variant):
    '''
    Create and return a dataloader for CIFAR10/100 and the epoch size
    '''
    TEST_BATCH_SIZE = 100

    if loader == 'DALI': # Using NVIDIA DALI
        raise NotImplementedError
    else: #using PyTorch default dataloader
        if loader_type is 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean[cifar_variant], std[cifar_variant]),
            ])

            if(cifar_variant == 'cifar10'):
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
                nClasses = 10
            elif(cifar_variant == 'cifar100'):
                trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
                nClasses = 100
            in_shape = [3, 32, 32]

            dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
            return dataloader, len(trainset)
        else: 
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean[cifar_variant], std[cifar_variant]),
            ])
            if(cifar_variant == 'cifar10'):
                testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
                nClasses = 10
            elif(cifar_variant == 'cifar100'):
                testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
                nClasses = 100
            in_shape = [3, 32, 32]

            dataloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
            return dataloader, len(testset)

def get_loader_imagenet(loader, loader_type, args):
    '''
    Create and return a dataloader for ImageNet and the epoch size.
    '''
    if loader == 'DALI': # Using NVIDIA DALI
        raise NotImplementedError
    else: #using PyTorch default dataloader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if loader_type is 'train':
            traindir = os.path.join(args.data, 'train')
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.87, contrast=0.5,
                                        saturation=0.5, hue=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
            return dataloader, len(trainset)
        else: 
            valdir = os.path.join(args.data, 'val')
            dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            return dataloader