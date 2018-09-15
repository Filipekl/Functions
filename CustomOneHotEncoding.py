class OneHotEncoder:

    def __init__(self):
        """Custom one hot encoder which is a wrapper around pandas get_dummies function."""
        self._cols = None
        self._classes = set()

    def fit(self, X, cols):
        """Generate classes for one hot encoder.

        :param X: pandas DataFrame
        :param cols: list Columns for which to perform one hot encoding
        """
        self._cols = cols
        _, self._classes = self._get_classes(X, self._cols)

    def transform(self, X):
        """Generate new columns for one hot encoder.

        :param X: pandas DataFrame
        """
        if self._classes:
            data, classes = self._get_classes(X, self._cols)

            add_cols = list(set(self._classes) - set(classes))
            for col in add_cols:
                data[col] = 0

            drop_cols = set(classes) - set(self._classes)
            data.drop(drop_cols, axis=1, inplace=True)
            return data

    def fit_transform(self, X, cols):
        self.fit(X, cols)
        return self.transform(X)

    @staticmethod
    def _get_classes(X, cols):
        data = pd.get_dummies(X, prefix_sep='__', columns=cols, dummy_na=True)

        classes = set()
        for col in cols:
            classes.update(data.columns[data.columns.str.contains(col)])

        return data, classes


