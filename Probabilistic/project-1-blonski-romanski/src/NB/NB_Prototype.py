from tqdm.auto import tqdm
import torch
import pyro
from pyro import distributions as dist
from torch.distributions import constraints
from collections import defaultdict
from sklearn import metrics as sk_mtr
class NaiveBayes:
    def __init__(self, lr,num_epochs):
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self._c_logits = None
        #probs
        self._num_probs = None
        self._cat_probs = None
        self.history = None

    def _get_log_likelihood(self, X):
        log_lk = []  # lista log_likelihoodów

        for c in range(self._num_cls):  # dla każdej klasy
            # dla numerycznych
            lps = []
            lps.extend([
                dist.Normal(v['mu'][c], v['sigma'][c]).log_prob(torch.tensor(X[nc].values)).long()
                for nc, v in self._num_probs.items()  # dla każdej cechy numerycznej
            ])
            # dla categorycznych
            lps.extend([
                dist.Categorical(v['probability'][c]).log_prob(torch.tensor(X[nc].values)).long()
                for nc, v in self._cat_probs.items()  # nc - name_column v-słownik z parametrami
            ])

            log_lk.append(torch.stack(lps).sum(dim=0))

        return torch.stack(log_lk).t()

    def _observe_classes(self,X,Y):
        if Y is not None:
            Y = torch.tensor(Y.values)
        log_lk = self._get_log_likelihood(X)
        log_pcx = pyro.deterministic('logP(c|x)', self._c_logits.log() + log_lk)
        with pyro.plate('data-pred', X.shape[0]):
            pyro.sample('c',dist.Categorical(logits=log_pcx),obs=Y,)

    def _init_c_logits(self):
        self._c_logits = pyro.param(
            'c_logits', torch.ones(self._num_cls).div(self._num_cls),constraint=constraints.simplex,)

    def _init_cat_params(self, X, categorical_cols):
        self._cat_probs = {
            col: {
                'probability': pyro.param(f'{col}_probability', torch.ones(self._num_cls,X[col].nunique()),constraint=constraints.simplex),
            }
            for col in categorical_cols
        }

    def _init_num_params(self, X, numerical_cols):
        self._num_probs = {  # tworzenie paramtrów odpowiedzialnych za rozkłady gaussianów dla każdej z K klas i N cech
            col: {
                'mu': pyro.param(f'{col}_mu', torch.zeros(self._num_cls,dtype=torch.float64)),  # num_cls to nasze K we wzorze.
                'sigma': pyro.param(f'{col}_sigma', torch.ones(self._num_cls,dtype=torch.float64)),
            }
            for col in numerical_cols  # numberical_cols to nasze N we wzorze up.Czyli N cech numerycznych
        }

    def _obs_categorical_features_given_classes(self, X, y):
        for c in range(self._num_cls):
            x_c = X[y==c] # weź dane których klasa odpowiada iteratowi z powyżej
            with pyro.plate(f'data-categorical-{c}', x_c.shape[0]):  #c - labelka klasy
                for nc, v in self._cat_probs.items(): #nc - name column v parametr dla rozkładu categorycznego
                    pyro.sample(
                        f'P(x_{nc}|c={c})',
                        dist.Categorical(v['probability'][c]),
                        obs=torch.tensor(x_c[nc].values)
                    )

    def _obs_numerical_features_given_classes(self, X, y):
        for c in range(self._num_cls):
            x_c = X[y == c]
            with pyro.plate(f'data-numerical-{c}', x_c.shape[0]):
                for nc, v in self._num_probs.items():  # nc-name_column, v-słownik z mu i sigma
                    pyro.sample(
                        f'P(x_{nc}|c={c})',
                        dist.Normal(v['mu'][c], v['sigma'][c]),
                        obs=torch.tensor(x_c[nc].values),
                    )

    def _model(self, X, y=None):

        if y is not None:  # training mode Gdy podamy wektor z klasami.
            self._num_cls = y.nunique() #number of unique classes
            categorical_cols = X.select_dtypes(include=['category'])
            numerical_cols = X.select_dtypes(include=['float64'])
            self._init_c_logits()
            self._init_num_params(X,numerical_cols)
            self._init_cat_params(X,categorical_cols)
            self._obs_categorical_features_given_classes(categorical_cols,y)
            self._obs_numerical_features_given_classes(numerical_cols,y)
        self._observe_classes(X, y)
        pass

    def _guide(self,X, y=None):
        pass #guide not needed

    def predict(self, X):
        pred = pyro.infer.Predictive(
            model=self._model,
            guide=self._guide,
            num_samples=1,
            return_sites=('logP(c|x)',),
        )
        log_pcx = pred(X)['logP(c|x)'].detach().squeeze(0).squeeze(0)
        y_pred = torch.argmax(log_pcx, dim=-1)
        return y_pred

    def test(self,test_X,test_Y):
        return sk_mtr.classification_report(y_true=test_Y, y_pred=self.predict(X=test_X),output_dict=True )

    def fit(self,data_X,data_Y):
        pyro.clear_param_store()
        svi = pyro.infer.SVI(
            model=self._model,
            guide=self._guide,
            optim=pyro.optim.Adam({'lr': self.learning_rate}),
            loss=pyro.infer.Trace_ELBO(),
        )

        self.history = {
            'losses': [],
            'params': defaultdict(list),
        }
        with tqdm(range(self.num_epochs)) as pbar:
            for epoch in pbar:
                loss = svi.step(data_X, data_Y)
                self.history['losses'].append(loss)

                p = dict(pyro.get_param_store())
                for k, v in p.items():
                    self.history['params'][k].append(v.detach().numpy().copy())

        return self.history

