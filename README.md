# bert_ned
Named Entity Disambiguation by training a BERT binary classification model.

The task that we will be trying to train BERT to solve is "Named Entity Dissambiguation". Namely, given an entity and different options of possible entities, and given the context of which the entity is being mentioned, which of the options in the context referring to?

For example, the entity "Jaguar" has a lot of options:

Jaguar as an animal
Jaguar as a brand of cars
Jaguar as a supercomputer
etc..

So, given the context:

"The man saw a Jaguar speed in the highway" -> the context is most likely referring to "Jaguar" as a car.
"The prey saw the jaguar cross the jungle" -> The context is most likely referring to "jaguar" as an animal.
We formulated this as a binary classification problem, giving BERT all entities on a dataset, as long as the context on which the entity is being mentioned, the option and the label 1/0 if it is correct of incorrect.

The dataset consists of all combinations of the different options for an entity and mentions in a sentence, along with a 0/1 label if the entity option is correct for a certain sentence mention (context). For more information on how this dataset was built to fine tune the model, see 1-make_dataset.ipynb notebook. Each "bert_qry" input text was shortened in order to avoid input texts that exceed 512 tokens.

For example, in this Jaguar example, the Data Set would look like:
```
]
    }
    "bert_qry": "Is 'Jaguar' in the context of: 'The man saw a Jaguar speed in the highway', referring to [SEP] Jaguar as an animal?",
    "label": 0,
    },
}
    "bert_qry": "Is 'Jaguar' in the context of: 'The man saw a Jaguar speed in the highway', referring to [SEP] Jaguar as a brand of cars?",
    "label": 1,
    },
}
    "bert_qry": "Is 'Jaguar' in the context of: 'The man saw a Jaguar speed in the highway', referring to [SEP] Jaguar as a supercomputer?",
    "label": 0,
    },
}
    "bert_qry": "Is 'Jaguar' in the context of: 'The prey saw the jaguar cross the jungle', referring to [SEP] Jaguar as an animal?",
    "label": 1,
    },
}
    "bert_qry": "Is 'Jaguar' in the context of: 'The prey saw the jaguar cross the jungle', referring to [SEP] Jaguar as a brand of cars?",
    "label": 0,
    },
}
    "bert_qry": "Is 'Jaguar' in the context of: 'The prey saw the jaguar cross the jungle', referring to [SEP] Jaguar as a supercomputer?",
    "label": 0,
    }
]
```

About the DataSet used for this training:

Our Dataset is comprised of news articles scrapped from a Mexican Newspaper. NER was applied to each article, so we got all Entities mentioned in the article. For each Entity, we performed a query to WikiData in order to extract all possible options for an Entity. An LLM was used as a teacher in order to disambiguate a subset of the whole dataset (0-ask_stable_beluga.ipynb) The teacher observations were used to get the labels. A preprocessing was performed in order to create the "bert_qry", making sure it doesn't exceed 512 tokens (1-make_dataset.ipynb).

`2-fine_tune_bert.ipynb` -> performs the Training phase of the BERT model.
`3-evaluate_bert_fine_tunning.ipynb` -> I created a class to perform Named Entity Disambiguation using my very own BERT NED model. I then use this class in order to perform Named Entity Disabiguation on a subset of the data set (a subset with which the model was not trained on) and compare it to the Disambiguation that our teacher performs (`Stable-Beluga-7B`).

The objective of fine tuning BERT is to have a lightweight model that can dismabiguate as good as Stable Belgua 7B (We don't need the whole power of an LLM if we are only performing a single task).

My NED model achieved an 86% accuracy against the teacher disambiguation abservations, it performs the disambiguation 3x faster that `Stable Beluga` does on a 4090 GPU and uses 3x less GPU memory than `Stable Beluga` LLM.