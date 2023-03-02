from src.eoe.training.bce import BCETrainer
from src.eoe.training.hsc import HSCTrainer
from src.eoe.training.dsvdd import DSVDDTrainer
from src.eoe.training.dsad import DSADTrainer
from src.eoe.training.focal import FocalTrainer
from src.eoe.training.clip import ADClipTrainer

TRAINER = {  # maps strings to trainer classes
    'hsc': HSCTrainer, 'bce': BCETrainer, 'clip': ADClipTrainer,
    'dsvdd': DSVDDTrainer, 'dsad': DSADTrainer, 'focal': FocalTrainer
}
