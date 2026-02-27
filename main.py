from args import get_args
import pandas as pd
import os
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader

def main():
    args = get_args()

    # 1. Read the dataframes
    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    val_df =  pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    # 2. Prepare datasets
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

if __name__ == "__main__":
    main()