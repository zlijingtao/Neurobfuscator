import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
# from HTMLParser import HTMLParser
class MaskedPCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, mask=None):
        # mask should contain selected cols. Suppose it is boolean to avoid code overhead
        self.n_components = n_components
        self.mask = mask
        self.pca = PCA(n_components=self.n_components)
    def fit(self, X, y = None):
        mask = self.mask
        mask = self.mask if self.mask is not None else slice(None)
        self.pca.fit(X[:, mask], y)
        return self

    def transform(self, X):
        mask = self.mask if self.mask is not None else slice(None)
        pca_transformed = self.pca.transform(X[:, mask])
        if self.mask is not None:
            remaining_cols = X[:, ~mask]
            return np.hstack([remaining_cols, pca_transformed])
        else:
            return pca_transformed
    def fit_transform(self, X, y=None):
        mask = self.mask if self.mask is not None else slice(None)
        pca_transformed = self.pca.fit_transform(X[:, mask], y)
        if self.mask is not None:
            remaining_cols = X[:, ~mask]
            print("total feature after PCA:")
            print(np.hstack([remaining_cols, pca_transformed]).shape[1])
            return np.hstack([remaining_cols, pca_transformed])
        else:
            print("total feature after PCA:")
            print(pca_transformed.shape[1])
            return pca_transformed
class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            print(feature_name)
            # if column_name == "Company" or column_name == "SourceName":
            #     fea = extractor.transform(X[column_name].values.astype('U'))
            
            if column_name == ['LocationRaw','LocationNormalized']:
                fea = extractor.transform(X['LocationRaw'],X['LocationNormalized'])
            else:
                fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]

    def fit_transform(self, X, y=None):
        for col in X.columns:
            print(col)
        extracted = []
        for feature_name, column_name, extractor in self.features:
            print(feature_name)
            # if column_name == "Company" or column_name == "SourceName":
            #     fea = extractor.fit_transform(X[column_name].values.astype('U'), y)
            if column_name == ['LocationRaw','LocationNormalized']:
                fea = extractor.transform(X['LocationRaw'],X['LocationNormalized'])
            else:
                fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            print("total feature:")
            print(np.concatenate(extracted, axis=1).shape[1])
            return np.concatenate(extracted, axis=1)
        else:
            print("total feature:")
            print(extracted[0].shape[1])
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T
