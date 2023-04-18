import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date

fake = Faker()
Faker.seed(12345)
random.seed(12345)

class SpecialTokens:
    SOS_token = '>'
    EOS_token = '<'
    UNK_token = '?'
    PAD_token = ';'


# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        m = m
        sme = SpecialTokens.SOS_token + m + SpecialTokens.EOS_token
        if h is not None:
            dataset.append((h, sme))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    dataset.sort(key= lambda x: len(x[0]))

    human_special_tokens = [SpecialTokens.PAD_token, SpecialTokens.UNK_token]
    machine_special_tokens = [SpecialTokens.PAD_token, SpecialTokens.SOS_token, SpecialTokens.EOS_token]
    
    human = dict(zip(human_special_tokens + sorted(human_vocab), 
                     list(range(len(human_vocab) + len(human_special_tokens)))))
    
    machine = dict(zip(machine_special_tokens + sorted(machine_vocab),
                    list(range(len(machine_vocab) + len(machine_special_tokens)))))
     
    return dataset, human, machine


