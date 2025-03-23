from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import sleep
import traceback
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm, trange
from PIL import Image
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml

from data.data_reader import ItemProcessor

logger = logging.getLogger(__name__)


class DataBriefReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class DataNoReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class MyDataset(Dataset):
    def __init__(self, config_path, item_processor: ItemProcessor, use_cache=True):
        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        self.MONGODB_URI = self.config["MONGODB_URI"]
        self.DATA_HOME = Path(self.config["DATA_HOME"])

        self.dataset = None
        for meta in self.config["META"]:
            if self.dataset is None:
                self.dataset = self.get_dataset(meta["collection"], use_cache=use_cache)
            else:
                self.dataset = pd.concat([self.dataset, self.get_dataset(meta["collection"], use_cache=use_cache)])

        logger.info(f"total length: {len(self)}")

        self.item_processor = item_processor


    def __len__(self):
        return len(self.dataset)


    def get_item_func(self, index: int):
        item = self.dataset.iloc[index]
        return self.item_processor.process_item(item)


    def __getitem__(self, index):
        try:
            return self.get_item_func(index)
        except Exception as e:
            logger.info(
                f"Item {index} errored, annotation:\n"
                f"{self.dataset[index]}\n"
                f"Error:\n"
                f"{traceback.format_exc()}"
            )
            raise RuntimeError


    def get_dataset(self, collection, use_cache=True):
        rank = dist.get_rank()
        if use_cache and (self.DATA_HOME / "shuffled" / f'{collection}-rank-{rank}.parquet').exists():
            print(f"Loading {collection} from cache")
            return pd.read_parquet(self.DATA_HOME / "shuffled" / f'{collection}-rank-{rank}.parquet')
        else:
            if dist.get_rank() == 0:
                print(f"Loading {collection} from MongoDB, this may take a while...")
                query = {}
                projection = {
                    "_id": 1,
                    "source_id": 1,
                    "media_path": 1,
                    "width": 1,
                    "height": 1,
                    "caption": 1,
                    "source": 1,
                }

                client = MongoClient(self.MONGODB_URI)
                db = client['world_model']
                collection = db[collection]

                cursor = collection.find(
                    query,
                    projection,
                    batch_size=8192,
                    no_cursor_timeout=True,
                    max_time_ms=2400000,
                )

                dataset = []
                for doc in tqdm(cursor):
                    dataset.append(doc)

                self.save_dataset(dataset, self.DATA_HOME / "shuffled" / f'{collection}-rank-{rank}.parquet')
            dist.barrier()
            return pd.read_parquet(self.DATA_HOME / "shuffled" / f'{collection}-rank-{rank}.parquet')


    @staticmethod
    def save_dataset(dataset, path):
        # Assuming 'data' is your large list of dictionaries
        batch_size = 10000
        schema = None  # Will be inferred from the first batch

        for i in trange(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            df_batch = pd.DataFrame(batch)
            df_batch['_id'] = df_batch['_id'].map(lambda x: str(x))
            
            # For the first batch, create the schema and ParquetWriter
            if i == 0:
                table = pa.Table.from_pandas(df_batch, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(path, schema)
                writer.write_table(table)
            else:
                table = pa.Table.from_pandas(df_batch, schema=schema, preserve_index=False)
                writer.write_table(table)

        # Close the writer when done
        if 'writer' in locals():
            writer.close()
            print("Parquet file created successfully!")