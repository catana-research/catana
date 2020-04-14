# -*- coding: utf-8 -*-
"""Data store object

Handles the efficient storage and retrieval of data, including snapshots and metadata. Built upon the Arctic framework.
"""
from copy import copy
import arctic


class DataStore(object):

    def __init__(self, mongo_host='localhost'):
        self._conn = arctic.Arctic(mongo_host)
        self.db = None
        self.directory = ''
        self.workspace = None

    def connect(self, db=None):
        self.db = self._conn[db]
        obj = copy(self)
        return obj

    def create_workspace(self, name, lib_type=arctic.VERSION_STORE):
        if name not in self._conn.list_libraries():
            self._conn.initialize_library(name, lib_type=lib_type)

    # TODO: Turn into a property
    def set_workspace(self, workspace):
        self.workspace = workspace
        self.db = self._conn[workspace]

    def list_workspaces(self):
        return self._conn.list_libraries()

    def projects(self):
        """TODO: Write projects to a projects index directory"""
        paths = self.ls(filter_project=False)
        projects = list(set([path.split('.')[0] for path in paths]))
        return projects

    def ls(self, project=None, filter_project=True):
        if not self.workspace:
            dirs = self.list_workspaces()
        else:
            dirs = self.db.list_symbols()
        return dirs

        if filter_project and self.workspace:
            filtered_directories = [directory for directory in dirs if directory.split('.')[0] == self.workspace]
        else:
            filtered_directories = dirs

        # TODO: Strip top level, remove current path: a.b.c --> b if dir = 'a'

        return filtered_directories

    def write(self, data=None, name='', category='root', project=None, metadata=None):
        # project = project or self.project
        # path = self._create_path(project, category, name)
        path = name
        self.db.write(symbol=path, data=data, metadata=metadata)

    def read(self, name='', category='root', project=None):
        # project = project or self.project
        # path = self._create_path(project, category, name)
        path = name
        #return self.db.read(self._map_key(name))
        return self.db.read(path)

    def _create_path(self, project, category, item):
        return '.'.join([project, category, item])

    def _map_key(self, key):
        """TODO: Split str by '.'"""
        return key



    def __getitem__(self, name):
        # return self.read(name)
        obj = copy(self)
        obj.dir = name
        return self.read(name)


if __name__ == '__main__':

    from catana.core.timer import Timer

    import pandas as pd
    df = pd.DataFrame({'a': a})
    df.to_csv('T:/Data/a.csv')
    df.to_parquet('T:/Data/a.parquet')

    with Timer() as t:
        df = pd.read_csv('T:/Data/a.csv')

    with Timer() as t:
        df = pd.read_parquet('T:/Data/a.parquet')

    ds = DataStore()

    ds.create_workspace('random')
    ds.list_workspaces()

    ds.set_workspace('random')
    ds.ls()



    import numpy as np
    data = {}
    data['normal'] = np.random.normal(0, 1, 10000)
    data['binomial'] = np.random.binomial(5, 0.5, 10000)
    data['poisson'] = np.random.poisson(2.5, 10000)
    data['beta'] = np.random.beta(0.1, 0.1, 10000)

    import matplotlib.pyplot as plt

    plt.hist(data['beta'])
    plt.show()

    ds.db.wr

    ds.write(data='a', name='gaussian', metadata={'info': 'Standard normal gaussian'})
    ds.db.write('normal', data['normal'], metadata={'info': 'Standard normal gaussian'})
    ds.db.write('binomial', data['binomial'], metadata={'info': 'Binomial n=5, p=0.5'})
    ds.db.write('poisson', data['poisson'], metadata={'info': 'Poisson lambda=2.5'})
    ds.db.write('beta', data['beta'], metadata={'info': 'Beta alpha=0.1, beta=0.1'})
    ds.ls()

    ds['gaussian'].data

    with Timer() as t:
        ds['gaussian'].data

    db = DataStore().connect('catana')

    db.set_project('random')

    db.projects()



    db.ls()
