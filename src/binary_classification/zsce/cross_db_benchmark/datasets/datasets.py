from dataclasses import dataclass


@dataclass()
class SourceDataset:
    name: str
    osf: bool = True


@dataclass()
class Database:
    db_name: str
    _source_dataset: str = None
    _data_folder: str = None
    max_no_joins: int = 4
    scale: int = 1
    contain_unicode: bool = False

    @property
    def source_dataset(self) -> str:
        if self._source_dataset is None:
            return self.db_name
        return self._source_dataset

    @property
    def data_folder(self) -> str:
        if self._data_folder is None:
            return self.db_name
        return self._data_folder


# datasets that can be downloaded from osf and should be unzipped
source_dataset_list = [
    # original datasets
    SourceDataset('airline'),
    SourceDataset('imdb'),
    SourceDataset('ssb'),
    SourceDataset('tpc_h'),
    SourceDataset('tpc_ds'),

]

database_list = [
    # unscaled
    Database('airline', max_no_joins=5),
    Database('imdb'),
    Database('ssb', max_no_joins=3),
    Database('tpc_h', max_no_joins=5),
    Database('tpc_ds', max_no_joins=5),

]

ext_database_list = database_list + [Database('imdb_full', _data_folder='imdb')]
