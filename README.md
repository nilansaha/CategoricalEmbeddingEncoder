## Categorical Embedding Encoder

Encode categorical variables using learned Embeddings

### Quick Start

#### For Classification Tasks

```
import pandas as pd
from categorical_embedding_encoder import CategoricalEmbeddingEncoder

df = pd.DataFrame({'feature':[1,2,3,1,2,1,2], 'target': [1,0,0,1,0,1,1]})

X = df['feature']
y = df['target']

encoder = CategoricalEmbeddingEncoder(classification = True, feature_name = 'feature_A')
encoder.fit(X,y)
encoder.transform(X)
```

#### For Regression Tasks

```
import pandas as pd
from categorical_embedding_encoder import CategoricalEmbeddingEncoder

df = pd.DataFrame({'feature':[1,2,3,1,2,1,2], 'target': [2.3,0.1,5.0,1.2,2.2,1.5,1.7]})

X = df['feature']
y = df['target']

encoder = CategoricalEmbeddingEncoder(classification = False, feature_name = 'feature_A')
encoder.fit(X,y)
encoder.transform(X)
```



