from enum import Enum


class Datatype(Enum):  # __init__ method needs 1 required positional argument: 'value'
    INT = 'int'
    FLOAT = 'float'
    CATEGORICAL = 'categorical'  # if len(column.unique()) <= categorical_threshold, type is set to categorical
    STRING = 'string' #  in this project, Datatype.STRING is not used at all.
    STRING_FLOAT = 'string_float'
    MISC = 'misc'  # if len(column.unique()) > categorical_threshold, type is set to MISC, such that 'date', 'time' type for the original column in the database

    def __str__(self):
        return self.value
