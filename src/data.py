# !/usr/bin/env python3
from pathlib import Path
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import random
import os

random.seed(100)

def load_image(path_image: str) -> Image.Image:
    """Load image from harddrive and return 3-channel PIL image.
    Args:
        path_image (str): image path
    Returns:
        Image.Image: loaded image
    """
    return Image.open(path_image).convert('RGB').resize((112,112), 4)


def get_person_image_paths(path_person_set: str) -> dict:
    """Creates mapping from person name to list of images.
    Args:
        path_person_set (str): Path to dataset that contains folder of images.
    Returns:
        Dict[str, List]: Mapping from person name to image paths,
                         For instance {'name': ['/path/image1.jpg', '/path/image2.jpg']}
    """
    # path_person_set = Path(path_person_set)
    # person_paths = filter(Path.is_dir, path_person_set.glob('*'))
    person_paths=[]
    for person_path in os.listdir(path_person_set):
      if '.ipynb' not in person_path:
        person_paths.append(Path(path_person_set+'/'+person_path))
    return {
        path.name:list(str(path)+'/'+file for file in list(os.listdir(path))) for path in person_paths
    }


def get_persons_with_at_least_k_images(person_paths: dict, k: int) -> list:
    """Filter persons and return names of those having at least k images
    Args:
        person_paths (dict): dict of persons, as returned by `get_person_image_paths`
        k (int): number of images to filter for
    Returns:
        list: list of filtered person names
    """
    return [name for name, paths in person_paths.items() if len(paths) >= k] 


class TripletFaceDataset(Dataset):

    def __init__(self, path, scale = 1, augment=False) -> None:
        super().__init__()

        self.scale = int(scale)

        self.person_paths = get_person_image_paths(path)
        self.persons = self.person_paths.keys()
        self.persons_positive = get_persons_with_at_least_k_images(self.person_paths, 2)

        if augment:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.GaussianBlur((5,5)),
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.4, 0.3, 0.2, 0.1),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

    def get_anchor_positive_negative_paths(self, index: int) -> tuple:
        """Randomly sample a triplet of image paths.
        Args:
            index (int): Index of the anchor / positive person.
        Returns:
            tuple[Path]: A triplet of paths (anchor, positive, negative)
        """

        # TODO Please implement this function
        
        
        # while len(list_anchor_img) == 0:
        if not self.persons_positive:
          raise ValueError("No positive persons available in the dataset.")
        # index = random.randint(0, len(self.persons_positive))
        # person = list(self.persons_positive)[index] # get the anchor person
        person = random.choice(list(self.persons_positive))
        list_person_img = self.person_paths[person][:] # images of the same person
        a_label = list(self.persons_positive).index(person)
        p_label = list(self.persons_positive).index(person)
        # print(list_person_img)
        list_anchor_img = []
       
        for anchor in list_person_img:
          if 'frame' not in os.path.basename(anchor.lower()):
            list_anchor_img.append(anchor)
            list_person_img.remove(anchor) # avoid taking the same image as positive 
          else:
            continue
        if not list_anchor_img:
          # Replace with a random image from the person's images
          a = random.choice(list_person_img)
        else:
          a = random.choice(list_anchor_img)  # get an anchor example

        # if not list_anchor_img:
        #   raise ValueError("No suitable anchor image found for person: " + person)
       
        p = random.choice(list_person_img) # get a positive example
        # get a negative person
        n_person = random.choice(list(self.persons_positive))
        while n_person == person:
            n_person = random.choice(list(self.persons_positive))
        n_label = list(self.persons_positive).index(n_person)
        list_nega_img = self.person_paths[n_person]
        list_nega_choose =[]
        for nega_img in list_nega_img:
          # print(nega_img)
          if 'frame' in os.path.basename(nega_img):
            list_nega_choose.append(nega_img)
          else:
            continue
        
        n = random.choice(list_nega_choose) # get a negative example
        return a, p, n, a_label, p_label, n_label


    def __getitem__(self, index: int):
        """Randomly sample a triplet of image tensors.
        Args:
            index (int): Index of the anchor / positive person.
        Returns:
            tuple[Path]: A triplet of tensors (anchor, positive, negative)
        """
        a, p, n, a_label, p_label, n_label = self.get_anchor_positive_negative_paths(index)
        # print(a,p,n,sep = '\n')
        return (
            self.transform(load_image(a)),
            self.transform(load_image(p)),
            self.transform(load_image(n)),
            a_label, p_label, n_label
        )

    def __len__(self):
        return self.scale * len(self.persons_positive)
    

class Triplet_loader(Dataset):
    def __init__(self, file_path, augment=False) -> None:
        super().__init__()
        with open(file_path,'r+') as f:
           triplet_paths = f.readlines()
           triplet_paths = [path.strip() for path in triplet_paths]
           self.triplet_paths = [path.split(' ') for path in triplet_paths]

    
        if augment:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.GaussianBlur((5,5)),
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.4, 0.3, 0.2, 0.1),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    def __getitem__(self, index: int):
        """Randomly sample a triplet of image tensors.
        Args:
            index (int): Index of the anchor / positive person.
        Returns:
            tuple[Path]: A triplet of tensors (anchor, positive, negative)
        """
        a, p, n, a_label, p_label, n_label = self.triplet_paths[index]
        
        # print(a_label, sep = '\n')

        return (
            self.transform(load_image(a)),
            self.transform(load_image(p)),
            self.transform(load_image(n)),
            int(a_label), int(p_label), int(n_label)
        )

    def __len__(self):

        return len(self.triplet_paths)




class TripletFaceDataset_new(Dataset):

    def __init__(self, path, scale = 1, augment=False) -> None:
        super().__init__()

        self.scale = int(scale)

        self.person_paths = get_person_image_paths(path)
        self.persons = self.person_paths.keys()
        self.persons_positive = get_persons_with_at_least_k_images(self.person_paths, 2)

        if augment:
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.GaussianBlur((5,5)),
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.4, 0.3, 0.2, 0.1),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

    def get_anchor_positive_negative_paths(self, index: int) -> tuple:
        """Randomly sample a triplet of image paths.
        Args:
            index (int): Index of the anchor / positive person.
        Returns:
            tuple[Path]: A triplet of paths (anchor, positive, negative)
        """

        # TODO Please implement this function
        
        
        # while len(list_anchor_img) == 0:
        if not self.persons_positive:
          raise ValueError("No positive persons available in the dataset.")
        # index = random.randint(0, len(self.persons_positive))
        # person = list(self.persons_positive)[index] # get the anchor person
        person = random.choice(list(self.persons_positive))
        list_person_img = self.person_paths[person][:] # images of the same person
        a_label = list(self.persons_positive).index(person)
        p_label = list(self.persons_positive).index(person)
        # print(list_person_img)
        list_anchor_img = []
       
        a = random.choice(list_person_img)
        list_person_img.remove(a) # avoid taking the same image as positive 
        
        # if not list_anchor_img:
        #   raise ValueError("No suitable anchor image found for person: " + person)
       
        p = random.choice(list_person_img) # get a positive example
        # get a negative person
        n_person = random.choice(list(self.persons_positive))
        while n_person == person:
            n_person = random.choice(list(self.persons_positive))
        n_label = list(self.persons_positive).index(n_person)
        list_nega_img = self.person_paths[n_person]
        n = random.choice(list_nega_img) # get a negative example
        return a, p, n, a_label, p_label, n_label


    def __getitem__(self, index: int):
        """Randomly sample a triplet of image tensors.
        Args:
            index (int): Index of the anchor / positive person.
        Returns:
            tuple[Path]: A triplet of tensors (anchor, positive, negative)
        """
        a, p, n, a_label, p_label, n_label = self.get_anchor_positive_negative_paths(index)
        # print(a,p,n,sep = '\n')
        return (
            self.transform(load_image(a)),
            self.transform(load_image(p)),
            self.transform(load_image(n)),
            a_label, p_label, n_label
        )

    def __len__(self):
        return self.scale * len(self.persons_positive)
    