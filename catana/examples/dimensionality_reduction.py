"""


"""
import pandas as pd
import altair as alt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from catana.plotting.altchart import AltChart

digits = datasets.load_digits(n_class=4)  # Restrict the number of classes
X = digits.data
y = digits.target
n_samples, n_features = X.shape

#X = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X)
df_pca = pd.DataFrame({'pca_1': X_pca[:, 0], 'pca_2': X_pca[:, 1], 'y': y})

df = pd.DataFrame(data=X_pca, columns=['pca1', 'pca2'])

selection1 = alt.selection(type='interval', resolve='global')
chart = AltChart(data=df_pca, size=(2, 2),
                 meta='2D PCA transform and marginal distributions'
    ).scatter(
        x='pca_1', y='pca_2', color='y:N', pos=(1, 0), interactive=True,
        brush=selection1,
        selection=selection1,
    ).histogram(
        x='pca_1', color='y:N', bins=40, pos=(0, 0), title='PCA_{1}',
        selection=selection1, stack=False, fill=False,
    ).histogram(
        x='pca_2', color='y:N', pos=(1, 1), title='PCA_{2}',
        selection=selection1, stack=False, fill=False,
    ).pie(
        column='pca_1', color='y', pos=(0, 1),
    )



estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)

model = pipe.fit(X, y)
prediction = model.predict(X)

df_pca['model_predict'] = prediction

selection2 = alt.selection_single(on='mouseover', nearest=True)
chart = AltChart(data=df_pca, size=(2, 2),
                 meta='2D PCA transform and marginal distributions'
    ).scatter(
        x='y', y='model_predict', color='y', pos=(0, 0), title='y vs prediction',
        selection=selection2,
    ).scatter(
        x='pca_1', y='pca_2', color='y:N', pos=(1, 0),
        selection=selection2,
        brush=selection2,
    ).line(
        x='pca_1', y='pca_2', color='y:N', pos=(1, 1), ci=True,
        selection=selection2,
        brush=selection2,
    )
