from autotrainer.autotrainer import AutoTrainer
import numpy as np

# Simulate image data
X_train = np.random.rand(5000, 3, 224, 224)  # 5000 images
y_train = np.random.randint(0, 10, 5000)     # 10 classes

trainer = AutoTrainer(X_train, y_train)
trainer.run()
