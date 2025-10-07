from torch.utils.data import DataLoader

from src.DepressionDataset import DepressionDataset

dataset = DepressionDataset(
    sessions_list=['300', '301'],
    root_dir='data/processed/sessions'
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
