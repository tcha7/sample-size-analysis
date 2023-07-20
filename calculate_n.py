import numpy as np
from scipy.special import erfinv
from scipy import stats

def cal_n(p, exp_diff, conf_level=0.95, type='ave_yn', data=None, max_iter=100):
    """
    calculate N based on:
    1. expected rate: `p` of which values are in [0,1]
    2. expected confidence interval: `exp_diff`
    3. confidence level: `conf_level`
    4. type of the statistics used, which may be one of the following:
        - 'ave_yn' is for when the statistic is the average of a bernoulli random variable, e.g. when you want to calc the average of clicks (since click event is a 0-1 random variable.) For this type, the standard deviation calculation will use the formula of Bernoulli's distribution.
        - 'ave_others' is when the statistic is the average of an unknown distribution. We will use a monte-carlo method to estimate sd.
    """
    
    z = _get_z(conf_level)
    if type == 'ave_yn':
        n = _cal_n_mean_bernoulli(p, exp_diff, z)
    elif type == 'ave_others':
        n = _cal_n_mean_monte_carlo(data, exp_diff, z, max_iter)
        
    return n

def cal_ci(data, conf_level=0.95):
    """
    cal confidence interval (ci) assuming that the statistic is normally distributed, e.g. mean.
    """
    z = _get_z(conf_level)
    n = len(data)
    sd = np.std(data)
    ci = z * sd / np.sqrt(n)
    return ci

def cal_n_from_t_test(p, exp_diff, conf_level=0.95, type='ave_yn', data=None, max_iter=100):
    """
    t-test (one-sample) is a hypothesis testing where the null hypothesis h0 is that the population mean is p. If t > threshold (e.g. ~1.65 for p-value < 5% with degree of freedom ~ 1000), we can reject n0 and conclude that the mean is actually x (or [p + exp_diff]). If t <= threshold, then the test is inconclusive (doesn't mean that the mean is p, because it means that the mean is either p or we simply don't have enough info to say it is not p.)
    calculate N from t-test. t-test considers |x - p| where p is the expected rate and |x - p| is the expected diff
    1. expected rate: `p` of which values are in [0,1]
    2. expected difference: `exp_diff`. 
    3. confidence level: `conf_level`
    4. type of the statistics used, which may be one of the following:
        - 'ave_yn' is for when the statistic is the average of a bernoulli random variable, e.g. when you want to calc the average of clicks (since click event is a 0-1 random variable.) For this type, the standard deviation calculation will use the formula of Bernoulli's distribution.
        - 'ave_others' is when the statistic is the average of an unknown distribution. We will use a monte-carlo method to estimate sd.
    """
    t = _get_t_stat(conf_level)
    if type == 'ave_yn':
        n = _cal_n_mean_bernoulli(p, exp_diff, t)
    elif type == 'ave_others':
        n = _cal_n_mean_monte_carlo(data, exp_diff, t, max_iter)
    return n    

def cal_ratio_for_comparing_two_means(total_n, exp_mean_1=None, exp_mean_2=None, conf_level=0.95, type='ave_yn', sd_1=None, sd_2=None):
    """
    Using two-sample t-test to work back and find what N1:N2 ratio we need to get a significant difference in means, where the total sample N = N1+N2 is given.
    If the rv is bernoulli (or y/n type of sample space), we can use `ave_yn`. Otherwise, use `ave_others`, which requires that we estimate sd and mean.
    """
    t = _get_t_stat(conf_level)
    if type == 'ave_yn':
        min_r, max_r = _cal_n_compare_two_means_bernoulli(mu1 = exp_mean_1, mu2 = exp_mean_2, total_n = total_n, t_lookup = t)
    elif type == 'ave_others':
        min_r, max_r = _cal_n_compare_two_means_general(exp_mean_1, exp_mean_2, sd_1, sd_2, total_n, t)
    
    print('current total_n = ' + str(total_n))
    if min_r == 0 and max_r == 0:
        print('Please increase `total_n`.')
    else:
        print(str(round(min_r,2)) + ' < sample ratio (r) < ' + str(round(max_r,2)) +',')
        print('where n1 = r*total_n and n2 = (1-r)*total_n')
    return min_r, max_r
    
    
def t_test(data, alpha=0.05, data2=None, base_mean=None, type='one-sample', method='stdlib'):
    """
     One-sample t-test is a hypothesis testing where the null hypothesis h0 is that the population mean is p. If t > threshold (e.g. ~1.65 for p-value < 5% with degree of freedom ~ 1000), we can reject n0 and conclude that the mean is actually x (or [p + exp_diff]). If t <= threshold, then the test is inconclusive (doesn't mean that the mean is p, because it means that the mean is either p or we simply don't have enough info to say it is not p.)
     Two-sample t-test checks whether mu1 differs from mu2, where the null hypothesis is that mu1 doesn't differ from mu2.
    """
    if method == 'stdlib': #use scipy-implemented codes
        if type == 'one-sample':
            t_stat, pvalue = stats.ttest_1samp(data, popmean = base_mean)    
        elif type == 'two-sample':
            t_stat, pvalue = stats.ttest_ind(data, data2) 
        
        print('t-statistic: ' + str(t_stat))
        print('p-value: ' + str(pvalue))
        
        if pvalue < alpha:
            print('The difference is significant.')
        else:
            print('The difference is inconclusive.')

    elif method == 'manual': #manual implementation
        if type == 'one-sample':
            t, t_lookup = _one_sample_t_test(data, base_mean, alpha)
        elif type == 'two-sample':
            t, t_lookup = _two_sample_t_test(data, data2, alpha)
        
        print('t-statistic: ' + str(t_stat))
        print('t-dist value: ' + str(t_lookup))
        
        if np.abs(t) > t_lookup:
            print('The difference is significant.')
        else:
            print('The difference is inconclusive.')
    else:
        print('method can either be `stdlib` or `manual`.')

def create_bernoulli_samples(p, n):
    """
    This is used to generate an array of {0,1} with the size n and the number of 1's around p*n.
    """
    data = np.zeros(int(n))
    num_bad = int(p * n)
    data[0:num_bad] = 1
    
    return data

def _get_z(conf_level):
    p = 1- (1-conf_level)/2
    z = np.sqrt(2)*erfinv(2*p-1)
    return z

def _get_t_stat(conf_level, df = None, n = None):
    if df is None and n is None:
        n = 100
        df = n - 1
    t_stat = stats.t.ppf(q=conf_level, df=df)
    return t_stat

def _cal_n_mean_bernoulli(p, exp_diff, z):
    n = p*(1-p)*((z/exp_diff)**2)
    return np.ceil(n)

def _cal_n_mean_monte_carlo(data, exp_diff, z, max_iter = 100):
    num_samp = 10.
    margin = 20
    iter_counter = 0
    error = margin+1
    num_iter = 100
    
    while np.abs(error) > margin:
        n = int(num_samp)
        num_samp = 0
        for i in range(num_iter):
            sub_data = np.random.choice(data, size=n)
            small_sd = np.std(sub_data)
            num_samp += (z*small_sd/exp_diff)**2
        num_samp /= num_iter
        error = num_samp - n

        iter_counter += 1
        if iter_counter >= max_iter:
            print('simulation reaches maximum iterations')
            break;
    return np.ceil(num_samp)

def _one_sample_t_test(data, base_mean, alpha=0.05):
    n = len(data)
    mu = np.mean(data)
    sd = np.std(data)
    t = (mu - base_mean) / (sd / np.sqrt(n))
    t_lookup = stats.t.ppf(q=1-alpha, df=n-1)
    return t, t_lookup

def _two_sample_t_test(data_1, data_2, alpha=0.05):
    """
    This two-sample t-test is implemented for when variances of the two samples are similar.
    If the variances are not assumed to be equal, use Welch's t-test using the lib stats.ttest_ind(equal_var=False) instead.
    """
    n1 = len(data_1)
    n2 = len(data_2)
    mu1 = np.mean(data_1)
    mu2 = np.mean(data_2)
    var1 = np.std(data_1)**2
    var2 = np.std(data_2)**2
    
    sp = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1 + n2 -2))
    t = (mu1 - mu2) / (sp * np.sqrt(1/n1 + 1/n2))
    t_lookup = stats.t.ppf(q=1-(alpha/2), df=n1+n2-2)
    
    return t, t_lookup

def _cal_n_compare_two_means_bernoulli(mu1, mu2, total_n, t_lookup):
    """
    Let n1 = r*n
        n2 = (1-r)*n
    Hence, n1:n2 = r:(1-r)
    """
    
    diff = mu1 - mu2
    alpha = (diff/t_lookup)**2
    var_1 = mu1 * (1 - mu1)
    var_2 = mu2 * (1 - mu2)
    
    a = total_n * alpha
    b = var_2 - var_1 - total_n*alpha
    c = var_1
    
    #check suffiiently large n
    if (b**2 - 4*a*c) < 0:
        print('The total n is not large enough.')
        min_r = 0
        max_r = 0
    else: 
        max_r = (-1*b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        min_r = (-1*b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    return min_r, max_r

def _cal_n_compare_two_means_general(mu1, mu2, sd_1, sd_2, total_n, t_lookup):
    """
    Let n1 = r*n
        n2 = (1-r)*n
    Hence, n1:n2 = r:(1-r)
    """
    
    expected_diff = mu1 - mu2
    alpha = (expected_diff / t_lookup)**2
    var_1 = sd_1**2
    var_2 = sd_2**2
    
    a = total_n * alpha
    b = var_2 - var_1 - total_n*alpha
    c = var_1
    
    #check suffiiently large n
    if (b**2 - 4*a*c) < 0:
        print('The total n is not large enough.')
        min_r = 0
        max_r = 0
    else: 
        max_r = (-1*b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        min_r = (-1*b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    return min_r, max_r
    