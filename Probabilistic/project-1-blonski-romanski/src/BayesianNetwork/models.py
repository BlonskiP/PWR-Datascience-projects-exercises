from pyro.infer import config_enumerate
import pyro
import torch
from pyro import distributions as dist
from torch.distributions import constraints

DEFAULT_ENUMERATE = 'sequential'

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_1(x):
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    churn_params = pyro.param('churn_params', torch.ones(3,3), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        churn = pyro.sample('churn', dist.Bernoulli(churn_params[internet, contract]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_2(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    churn_params = pyro.param('churn_params', torch.ones(2,2,2,3,3,3,3,3,3,2,4), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[senior, partner, dependents, internet, security, support, protection, backup, contract, paperless, payment]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_3_1(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z3_params = pyro.param('z3_params', torch.ones(3,3,3,3) / 2, constraint=constraints.interval(0,1))
    z4_params = pyro.param('z4_params', torch.ones(3,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,2,4,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z3 = pyro.sample('z3', dist.Bernoulli(z3_params[security, support, protection, backup])).long()
        z4 = pyro.sample('z4', dist.Bernoulli(z4_params[internet, z3])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, z4, contract, z5]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_3_2(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z3_params = pyro.param('z3_params', torch.ones(3,3,3,3) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,3,2,4,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z3 = pyro.sample('z3', dist.Bernoulli(z3_params[security, support, protection, backup])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, z3, contract, z5]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_3_3(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z3_params = pyro.param('z3_params', torch.ones(3,3,3,3) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,3,2,3,2,4), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z3 = pyro.sample('z3', dist.Bernoulli(z3_params[security, support, protection, backup])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, z3, contract, paperless, payment]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_3_4(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z3_params = pyro.param('z3_params', torch.ones(3,3,3,3) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,2,3,2,4,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z3 = pyro.sample('z3', dist.Bernoulli(z3_params[security, support, protection, backup])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z1, dependents, internet, z3, contract, z5]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_3_5(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    z3_params = pyro.param('z3_params', torch.ones(3,3,3,3) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,2,2,3,2,4,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        z3 = pyro.sample('z3', dist.Bernoulli(z3_params[security, support, protection, backup])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[senior, partner, dependents, internet, z3, contract, z5]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_3_6(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,3,3,3,3,3,4,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()

        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, protection, backup, contract, z5]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_4_1(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)
    tenure_params = pyro.param('tenure_params', torch.ones(3) / 3, constraint=constraints.simplex)
    charge_params = pyro.param('charge_params', torch.ones(3) / 3, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,3,3,3,3,3,4,2,3,3), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()
        tenure = pyro.sample('tenure', dist.Categorical(tenure_params), obs=x.Tenure).long()
        charge = pyro.sample('charge', dist.Categorical(charge_params), obs=x.MonthlyCharges).long()


        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, protection, backup, contract, z5, tenure, charge]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_4_2(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    protection_params = pyro.param('protection_params', torch.ones(3) / 3, constraint=constraints.simplex)
    backup_params = pyro.param('backup_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)
    tenure_params = pyro.param('tenure_params', torch.ones(3) / 3, constraint=constraints.simplex)
    charge_params = pyro.param('charge_params', torch.ones(3) / 3, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))
    z6_params = pyro.param('z6_params', torch.ones(3,3) / 2, constraint=constraints.interval(0,1))


    churn_params = pyro.param('churn_params', torch.ones(2,3,3,3,3,3,4,2,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        protection = pyro.sample('protection', dist.Categorical(protection_params), obs=x.DeviceProtection).long()
        backup = pyro.sample('backup', dist.Categorical(backup_params), obs=x.OnlineBackup).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()
        tenure = pyro.sample('tenure', dist.Categorical(tenure_params), obs=x.Tenure).long()
        charge = pyro.sample('charge', dist.Categorical(charge_params), obs=x.MonthlyCharges).long()


        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()
        z6 = pyro.sample('z6', dist.Bernoulli(z6_params[tenure, charge])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, protection, backup, contract, z5, z6]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_4_3(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)
    tenure_params = pyro.param('tenure_params', torch.ones(3) / 3, constraint=constraints.simplex)
    charge_params = pyro.param('charge_params', torch.ones(3) / 3, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))
    z6_params = pyro.param('z6_params', torch.ones(3,3) / 2, constraint=constraints.interval(0,1))


    churn_params = pyro.param('churn_params', torch.ones(2,3,3,3,4,2,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()
        tenure = pyro.sample('tenure', dist.Categorical(tenure_params), obs=x.Tenure).long()
        charge = pyro.sample('charge', dist.Categorical(charge_params), obs=x.MonthlyCharges).long()


        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()
        z6 = pyro.sample('z6', dist.Bernoulli(z6_params[tenure, charge])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, contract, z5, z6]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_5_1(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)
    tenure_params = pyro.param('tenure_params', torch.ones(3) / 3, constraint=constraints.simplex)
    charge_params = pyro.param('charge_params', torch.ones(3) / 3, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2,3) / 3, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(3,2,3) / 3, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4,3) / 3, constraint=constraints.interval(0,1))
    z6_params = pyro.param('z6_params', torch.ones(3,3,3) / 3, constraint=constraints.interval(0,1))


    churn_params = pyro.param('churn_params', torch.ones(3,3,3,3,4,3,3), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()
        tenure = pyro.sample('tenure', dist.Categorical(tenure_params), obs=x.Tenure).long()
        charge = pyro.sample('charge', dist.Categorical(charge_params), obs=x.MonthlyCharges).long()


        z1 = pyro.sample('z1', dist.Categorical(z1_params[senior, partner, :])).long()
        z2 = pyro.sample('z2', dist.Categorical(z2_params[z1, dependents, :])).long()
        z5 = pyro.sample('z5', dist.Categorical(z5_params[paperless, payment, :])).long()
        z6 = pyro.sample('z6', dist.Categorical(z6_params[tenure, charge, :])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, contract, z5, z6]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_5_2(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)
    tenure_params = pyro.param('tenure_params', torch.ones(3) / 3, constraint=constraints.simplex)
    charge_params = pyro.param('charge_params', torch.ones(3) / 3, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))
    z6_params = pyro.param('z6_params', torch.ones(3,3) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.zeros(2,3,3,3,4,2,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()
        tenure = pyro.sample('tenure', dist.Categorical(tenure_params), obs=x.Tenure).long()
        charge = pyro.sample('charge', dist.Categorical(charge_params), obs=x.MonthlyCharges).long()


        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()
        z6 = pyro.sample('z6', dist.Bernoulli(z6_params[tenure, charge])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, contract, z5, z6]), obs=x.y)
        return churn

@config_enumerate(default=DEFAULT_ENUMERATE)
def model_5_3(x):

    senior_params = pyro.param('senior_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    partner_params = pyro.param('partner_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    dependents_params = pyro.param('dependents_params', torch.tensor(0.5) / 2, constraint=constraints.interval(0,1))
    internet_params = pyro.param('internet_params', torch.ones(3) / 3, constraint=constraints.simplex)
    security_params = pyro.param('security_params', torch.ones(3) / 3, constraint=constraints.simplex)
    support_params = pyro.param('support_params', torch.ones(3) / 3, constraint=constraints.simplex)
    contract_params = pyro.param('contract_params', torch.ones(3) / 3, constraint=constraints.simplex)
    paperless_params = pyro.param('paperless_params', torch.tensor(0.5), constraint=constraints.interval(0,1))
    payment_params = pyro.param('payment_params', torch.ones(4) / 4, constraint=constraints.simplex)
    tenure_params = pyro.param('tenure_params', torch.ones(5) / 3, constraint=constraints.simplex)
    charge_params = pyro.param('charge_params', torch.ones(5) / 3, constraint=constraints.simplex)

    z1_params = pyro.param('z1_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z2_params = pyro.param('z2_params', torch.ones(2,2) / 2, constraint=constraints.interval(0,1))
    z5_params = pyro.param('z5_params', torch.ones(2,4) / 2, constraint=constraints.interval(0,1))
    z6_params = pyro.param('z6_params', torch.ones(5,5) / 2, constraint=constraints.interval(0,1))

    churn_params = pyro.param('churn_params', torch.ones(2,3,3,3,4,2,2), constraint=constraints.interval(0,1))

    with pyro.plate('data', x.shape[0]):
        senior = pyro.sample('senior', dist.Bernoulli(senior_params), obs=x.SeniorCitizen).long()
        partner = pyro.sample('partner', dist.Bernoulli(partner_params), obs=x.Partner).long()
        dependents = pyro.sample('dependents', dist.Bernoulli(dependents_params), obs=x.Dependents).long()
        internet = pyro.sample('internet', dist.Categorical(internet_params), obs=x.InternetService).long()
        security = pyro.sample('security', dist.Categorical(security_params), obs=x.OnlineSecurity).long()
        support = pyro.sample('support', dist.Categorical(support_params), obs=x.TechSupport).long()
        contract = pyro.sample('contract', dist.Categorical(contract_params), obs=x.Contract).long()
        paperless = pyro.sample('paperless', dist.Bernoulli(paperless_params), obs=x.PaperlessBilling).long()
        payment = pyro.sample('payment', dist.Categorical(payment_params), obs=x.PaymentMethod).long()
        tenure = pyro.sample('tenure', dist.Categorical(tenure_params), obs=x.Tenure).long()
        charge = pyro.sample('charge', dist.Categorical(charge_params), obs=x.MonthlyCharges).long()


        z1 = pyro.sample('z1', dist.Bernoulli(z1_params[senior, partner])).long()
        z2 = pyro.sample('z2', dist.Bernoulli(z2_params[z1, dependents])).long()
        z5 = pyro.sample('z5', dist.Bernoulli(z5_params[paperless, payment])).long()
        z6 = pyro.sample('z6', dist.Bernoulli(z6_params[tenure, charge])).long()

        churn = pyro.sample('churn', dist.Bernoulli(churn_params[z2, internet, security, support, contract, z5, z6]), obs=x.y)
        return churn