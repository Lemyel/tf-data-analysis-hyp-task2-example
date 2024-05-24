import pandas as pd
import numpy as np


chat_id = 280785885 # Ваш chat ID, не меняйте название переменной

def solution(sample1: np.ndarray, sample2: np.ndarray) -> bool:
    
    n1, n2 = len(sample1), len(sample2)
    
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    
    t_stat = (mean1 - mean2) / se
    
    df = (std1**2 / n1 + std2**2 / n2)**2 / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
    
    p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
    
    return p_value < 0.05



