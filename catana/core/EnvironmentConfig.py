"""


"""

#
#
# frozendict(**kwargs):
# return frozendata.frozendict(kwargs)
#
# class ProjectConfig(object):
#     pass
#
# class Config(object):
#     pass
#
# CONFIG = ProjectConfig(
#     project='catana',
#
#     base=Config(
#         desc='Base configuration',
#
#         db=frozendict(
#             mongo_host='localhost',
#             name='',
#         ),
#         dask=frozendict(
#             max_concurrent=1,
#         ),
#         dir=frozendict(
#             a='a',
#         )
#     ),
#
#     user=Config(
#             desc='User configuration',
#             base='base',
#             dir=frozendict(
#                 a='b',
#             )
#         ),
#
#
#
#
# )

class EnvironmentConfig(object):

    def __init__(self, project, **environments):

        self. project
        self._env = project
        self._environments = dict()

        for tag, e in environments.items()
