# bert_ned
Named Entity Disambiguation by training a BERT binary classification model.

The task that we will be trying to train BERT to solve is "Named Entity Dissambiguation". Namely, given an entity and different options of possible entities, and given the context of which the entity is being mentioned, which of the options in the context referring to?

For example, the entity "Jaguar" has a lot of options:

Jaguar as an animal
Jaguar as a brand of cars
Jaguar as a supercomputer
etc..

So, given the context:

- "The man saw a Jaguar speed in the highway" -> the context is most likely referring to "Jaguar" as a car.

- "The prey saw the jaguar cross the jungle" -> The context is most likely referring to "jaguar" as an animal.

About the DataSet used for this training:

The dataset consists of news articles obtained from a Mexican newspaper, processed using Named Entity Recognition (NER) to identify entities within each article. Queries were made to WikiData for each identified entity in order to gather all potential matches of an entity. The `StableBeluga-7B` Language Model (LLM) assisted in disambiguating selected entities from the dataset in `0-ask_stable_beluga.ipynb`, with its outputs serving as labels for training.

This project approaches the task as a binary classification problem. The training data includes entities from the articles, relevant sentences (context) where the entity is being mentioned and all WikiData options. Each entity-context-option triplet was paired with a binary label (1/0) to form a single training observation. The dataset construction process aimed to fine-tune the model, as detailed in the `1-make_dataset.ipynb` notebook. To ensure compatibility with model limitations, inputs were truncated to fit within a 512-token maximum.

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


Notebooks description:
- `0-ask_stable_beluga.ipynb` -> Create the teacher observations by triggering `StableBeluga-7B`.
- `1-make_dataset.ipynb` -> Pre process Data set to make it available for BERT fine tunning.
- `2-fine_tune_bert.ipynb` -> performs the Training phase of the BERT model.
- `3-evaluate_bert_fine_tunning.ipynb` -> I created a class to perform Named Entity Disambiguation using my very own BERT NED model. I then use this class in order to perform Named Entity Disabiguation on a subset of the data set (a subset with which the model was not trained on) and compare it to the Disambiguation that our teacher performs (`Stable-Beluga-7B`).

The objective of fine tuning BERT is to have a lightweight model that can dismabiguate as good as Stable Belgua 7B (We don't need the whole power of an LLM if we are only performing a single task).

My NED model achieved an 86% accuracy against the teacher disambiguation abservations, it performs the disambiguation 3x faster that `Stable Beluga` does on a 4090 GPU and uses 3x less GPU memory than `Stable Beluga` LLM.