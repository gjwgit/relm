import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt

from mlhub.pkg import mlask, mlcat
from IPython.display import display
from collections import Counter

from relm.mechanisms import LaplaceMechanism


mlcat("Differentially Private Release Mechanism", """\
This demo is based on the Jupyter Notebook from the RelM
package on github.

RelM can be readily utilised for the differentially private
release of data. In our demo database the records indicate the age
group of each patient who received a COVID-19 test on 9 March 2020.
Each patient is classified as belonging to one of eight age groups:
0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, and 70+.  One common
way to summarise this kind of data is with a histogram.  That is, to
report the number of patients that were classified as belonging to
each age group.

For this demonstration we will create a histogram for the actual data
and then a histogram for differentially private data.

The data is first loaded from a csv file. It simply consists of two
columns, the first is the date and the scond is the age group.""")

# Read the raw data.

data = pd.read_csv("pcr_testing_age_group_2020-03-09.csv")

mlask(True, True)

# Compute the exact query responses.

exact_counts = data["age_group"].value_counts().sort_index()
values = exact_counts.values

mlcat("Data Sample", """\
Here's a random sample of some of the records:
""")

print(data.sample(10))

mlask(True, True)

mlcat("Laplace Mechanism", """\
The Laplace mechanism can be used to produce a differentially private
histogram that summarises the data without compromising the privacy of
the patients whose data comprise the database. To do so, Laplace noise
is added to the count for each age group and the noisy counts are
released instead of the exact counts.

The noise that is added results in perturbed values that are real
numbers rather than integers, and so if the results we are expecting
are integers, the results can be rounded without loss of privacy. We
do that here for our peturbed values.""")

# Create a differentially private release mechanism

epsilon = 0.1
mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=1.0)
perturbed_counts = mechanism.release(values=values.astype(np.float))
perturbed_counts = perturbed_counts.astype(np.int64)

mlask(True, True)

mlcat("Choosing Epsilon", f"""\
The magnitude of the differences between the exact counts
and perturbed counts depends only on the value of the privacy parameter,
epsilon. Smaller values of epsilon yield larger perturbations. Larger
perturbations yeild lower utility.

To understand this compare the actual and peturbed values below. If
the value of epsilon is not too small, then we expect that the two
histograms will look similar.

For our purposes we have chosen epsilon as {epsilon} resulting in the
following peturbation.
""")

# Extract the set of possible age groups.

age_groups = np.sort(data["age_group"].unique())

# Reformat the age group names for nicer display.

age_ranges = np.array([a.lstrip("AgeGroup_") for a in age_groups])

# Create a dataframe with both exact and perturbed counts.

column_names = ["Age Group", "Exact Counts", "Perturbed Counts"]
column_values = [age_ranges, values, perturbed_counts]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table)

# Display as a table.

print(df)

mlask(True, True)

mlcat("Visualising the Perturbations", """\
The two histograms show that the peturbed values remain consistent
with the true values, whilst ensuring privacy.
""")

# Plot the two histograms as bar graphs.

df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
plt.show()

exit()

mlcat("Geometric Mechanism", """\

TODO

In this example, all of the exact counts are integers. That is because
they are the result of so-called counting queries. The perturbed counts
produced by the Laplace mechanism are real-valued. In some applications,
e.g. when some downstream processing assumes it will receive
integer-valued data, we may need the perturbed counts to be integers.
One way to achieve this is by simply rounding the outputs of the Laplace
mechanism to the nearest integer. Because this differentially private
release mechanisms are not affected by this kind of post-processing,
doing so will not affect any privacy guarantees.

Alternatively, we could use the geometric mechanism to compute the
permuted counts. The geometric mechanism is simply a discrete version of
the Laplace mechanism and it produces integer valued perturbations.
""")
mlask()

mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
# Create a differentially private release mechanism
from relm.mechanisms import GeometricMechanism
mechanism = GeometricMechanism(epsilon=0.1, sensitivity=1.0)
perturbed_counts = mechanism.release(values=values)
""")
mlask()
# Create a differentially private release mechanism
from relm.mechanisms import GeometricMechanism
mechanism = GeometricMechanism(epsilon=0.1, sensitivity=1.0)
perturbed_counts = mechanism.release(values=values)
mlcat("", """Visualising the Results
As with the Laplace mechanism, we can plot the exact histogram alongside
the differentially private histogram to get an idea if we have used too
small a value for epsilon.
""")
mlask()

mlcat("", """
# Create a dataframe with both exact and perturbed counts
column_values = [age_ranges, values, perturbed_counts]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table)

# Display the two histograms as a table
display(df.style.set_caption("Test Counts by Age Group"))

# Plot the two histograms as bar graphs
df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
plt.show()
""")
mlask()
# Create a dataframe with both exact and perturbed counts
column_values = [age_ranges, values, perturbed_counts]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table)

# Display the two histograms as a table
display(df.style.set_caption("Test Counts by Age Group"))

# Plot the two histograms as bar graphs
df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
plt.show()
mlcat("", """Exponential Mechanism
""")
mlask()

mlcat("", """Basic Usage
The ExponentialMechanism does not lend itself to vectorised queries as
easily as the LaplaceMechanism or GeometricMechanism. So, to produce a
histogram query that is comparable to those discussed above we wrap the
query releases in a loop and compute them one at a time.
""")
mlask()

mlcat("", """
# Create a differentially private release mechanism
from relm.mechanisms import ExponentialMechanism

output_range = np.arange(2**10)
utility_function = lambda x: -abs(output_range - x)

perturbed_counts = np.empty(len(values), dtype=np.int)
for i, value in enumerate(values.astype(np.float)):
    mechanism = ExponentialMechanism(epsilon=0.1,
                                     utility_function=utility_function,
                                     sensitivity=1.0,
                                     output_range=output_range)
    
    perturbed_counts[i] = mechanism.release(values=value)
""")
mlask()
# Create a differentially private release mechanism
from relm.mechanisms import ExponentialMechanism

output_range = np.arange(2**10)
utility_function = lambda x: -abs(output_range - x)

perturbed_counts = np.empty(len(values), dtype=np.int)
for i, value in enumerate(values.astype(np.float)):
    mechanism = ExponentialMechanism(epsilon=0.1,
                                     utility_function=utility_function,
                                     sensitivity=1.0,
                                     output_range=output_range)
    
    perturbed_counts[i] = mechanism.release(values=value)
mlcat("", """Visualising the Results
""")
mlask()

mlcat("", """
# Create a dataframe with both exact and perturbed counts
column_values = [age_ranges, values, perturbed_counts]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table)

# Display the two histograms as a table
display(df.style.set_caption("Test Counts by Age Group"))

# Plot the two histograms as bar graphs
df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
plt.show()
""")
mlask()
# Create a dataframe with both exact and perturbed counts
column_values = [age_ranges, values, perturbed_counts]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table)

# Display the two histograms as a table
display(df.style.set_caption("Test Counts by Age Group"))

# Plot the two histograms as bar graphs
df.plot(x="Age Group", title="Test Counts by Age Group", kind="bar", rot=0)
plt.show()
mlcat("", """Sparse Mechanisms
We currently have four mechanisms that take advantage of sparsity to
answer more queries about the data for a given privacy budget. All of
these mechanisms compare noisy query responses to a noisy threshold
value. If a noisy response does not exceed the noisy threshold, then the
mechanism reports only that the value did not exceed the threshold.
Otherwise, the mechanism reports that the value exceeded the threshold.
Furthermore, in the latter case some mechanisms release more information
about the underlying exact count. This extra information is computed
using some other differentially private mechanism and therefore imposes
some additional privacy costs.
""")
mlask()

mlcat("", """Data Wrangling
All three of our mechanims share an input format. We require a sequence
of exact query responses and a threshold value to which these responses
will be compared.
""")
mlask()

mlcat("", """
# Read the raw data
fp = '20200811_QLD_dummy_dataset_individual_v2.xlsx'
data = pd.read_excel(fp)

# Limit our attention to the onset date column
data.drop(list(data.columns[1:]), axis=1, inplace=True)  

# Remove data with no onset date listed
mask = data['ONSET_DATE'].notna()
data = data[mask]

# Compute the exact query responses
queries = [(pd.Timestamp('2020-01-01')  + i*pd.Timedelta('1d'),) for i in range(366)]
exact_counts = dict.fromkeys(queries, 0)
exact_counts.update(data.value_counts())

dates, values = zip(*sorted(exact_counts.items()))
values = np.array(values, dtype=np.float64)
""")
mlask()
# Read the raw data
fp = '20200811_QLD_dummy_dataset_individual_v2.xlsx'
data = pd.read_excel(fp)

# Limit our attention to the onset date column
data.drop(list(data.columns[1:]), axis=1, inplace=True)  

# Remove data with no onset date listed
mask = data['ONSET_DATE'].notna()
data = data[mask]

# Compute the exact query responses
queries = [(pd.Timestamp('2020-01-01')  + i*pd.Timedelta('1d'),) for i in range(366)]
exact_counts = dict.fromkeys(queries, 0)
exact_counts.update(data.value_counts())

dates, values = zip(*sorted(exact_counts.items()))
values = np.array(values, dtype=np.float64)
mlcat("", """Above Threshold
The simplest of the sparse release mechanisms simply reports the index
of the first query response that exceeds the specified threshold.
""")
mlask()

mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
from relm.mechanisms import AboveThreshold
mechanism = AboveThreshold(epsilon=1.0, sensitivity=1.0, threshold=100)
index = mechanism.release(values)
""")
mlask()
from relm.mechanisms import AboveThreshold
mechanism = AboveThreshold(epsilon=1.0, sensitivity=1.0, threshold=100)
index = mechanism.release(values)
mlcat("", """Empirical Distribution
Because AboveThreshold returns a single index, we can run many
experiments and plot a histogram of the results of those experiments to
get an empirical estimate of the distribution of the mechanism’s output.
""")
mlask()

mlcat("", """
TRIALS = 2**10
threshold = 100
results = np.zeros(TRIALS)
for i in range(TRIALS):
    mechanism = AboveThreshold(epsilon=1.0, sensitivity=1.0, threshold=threshold)
    index = mechanism.release(values)
    results[i] = index    
""")
mlask()
TRIALS = 2**10
threshold = 100
results = np.zeros(TRIALS)
for i in range(TRIALS):
    mechanism = AboveThreshold(epsilon=1.0, sensitivity=1.0, threshold=threshold)
    index = mechanism.release(values)
    results[i] = index    
mlcat("", """Visualising the Results
""")
mlask()

mlcat("", """
print("The index of the first exact count that exceeds the threshold is: %i\n" % np.argmax(values >= threshold))

histogram = dict.fromkeys(np.arange(min(results), max(results)), 0)
histogram.update(Counter(results))

plt.xlabel("Index")
plt.ylabel("Count")
plt.title("Distribution of AboveThreshold outputs")
plt.plot(list(histogram.keys()), list(histogram.values()))
plt.axvline(x=np.argmax(values >= threshold), color='orange')
""")
mlask()
print("The index of the first exact count that exceeds the threshold is: %i\n" % np.argmax(values >= threshold))

histogram = dict.fromkeys(np.arange(min(results), max(results)), 0)
histogram.update(Counter(results))

plt.xlabel("Index")
plt.ylabel("Count")
plt.title("Distribution of AboveThreshold outputs")
plt.plot(list(histogram.keys()), list(histogram.values()))
plt.axvline(x=np.argmax(values >= threshold), color='orange')
mlcat("", """Sparse Indicator
The SparseIndicator is a straightforward extension of the AboveThreshold
mechanism. Here, we find the indices of several values that exceeds the
specified threshold. The number of indices that this mechanism will
return is controlled by the cutoff parameter.
""")
mlask()

mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
from relm.mechanisms import SparseIndicator
mechanism = SparseIndicator(epsilon=1.0, sensitivity=1.0, threshold=100, cutoff=3)
indices = mechanism.release(values)
""")
mlask()
from relm.mechanisms import SparseIndicator
mechanism = SparseIndicator(epsilon=1.0, sensitivity=1.0, threshold=100, cutoff=3)
indices = mechanism.release(values)
mlcat("", """Visualising the Results
""")
mlask()

mlcat("", """
TRIALS = 16
cutoff = 4
threshold = 100
indices = np.empty((TRIALS, cutoff), dtype=np.int)
for i in range(TRIALS):
    mechanism = SparseIndicator(epsilon=1.0, sensitivity=1.0, threshold=threshold, cutoff=cutoff)
    indices[i] = mechanism.release(values)
    
df = pd.DataFrame(indices,
                  columns=["Hit %i" % (j+1) for j in range(cutoff)],
                  index=["Trial %i" % i for i in range(TRIALS)])
display(df)
""")
mlask()
TRIALS = 16
cutoff = 4
threshold = 100
indices = np.empty((TRIALS, cutoff), dtype=np.int)
for i in range(TRIALS):
    mechanism = SparseIndicator(epsilon=1.0, sensitivity=1.0, threshold=threshold, cutoff=cutoff)
    indices[i] = mechanism.release(values)
    
df = pd.DataFrame(indices,
                  columns=["Hit %i" % (j+1) for j in range(cutoff)],
                  index=["Trial %i" % i for i in range(TRIALS)])
display(df)
mlcat("", """Sparse Numeric
The SparseNumeric mechanism returns perturbed values alongside the
indices of the values that exceeded the threshold.
""")
mlask()

mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
from relm.mechanisms import SparseNumeric
mechanism = SparseNumeric(epsilon=1.0, sensitivity=1.0, threshold=100, cutoff=3)  
indices, perturbed_values = mechanism.release(values)
""")
mlask()
from relm.mechanisms import SparseNumeric
mechanism = SparseNumeric(epsilon=1.0, sensitivity=1.0, threshold=100, cutoff=3)  
indices, perturbed_values = mechanism.release(values)
mlcat("", """Visualising the Results
Notice that in these experiemnts we have set epsilon=4.0. This is a
larger value than we use in the other sparse mechanisms. Because the
SparseNumeric mechanism releases more information about the underlying
exact query responses than does SparseIndices, for example, it consumes
the available privacy budget more quickly. To achieve comparable utility
with respect the the indices returned by the two mechanisms, we
therefore need to a larger value for epsilon.
""")
mlask()

mlcat("", """
TRIALS = 2**4
cutoff = 3
threshold = 100
indices = np.empty(shape=(TRIALS, cutoff), dtype=np.int)
perturbed_values = np.empty(shape=(TRIALS, cutoff), dtype=np.float)
for i in range(TRIALS):
    mechanism = SparseNumeric(epsilon=4.0, sensitivity=1.0, threshold=threshold, cutoff=cutoff)  
    indices[i], perturbed_values[i] = mechanism.release(values)

hit_names = ["Hit %i" % (j+1) for j in range(cutoff)]
value_names = ["Value %i" % (j+1) for j in range(cutoff)]
column_pairs = zip(hit_names, value_names)
column_names = [val for pair in column_pairs for val in pair]
value_pairs = zip(indices.transpose(), perturbed_values.transpose())
column_values = [val for pair in value_pairs for val in pair]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table, index=["Trial %i" % i for i in range(TRIALS)])
display(df)
""")
mlask()
TRIALS = 2**4
cutoff = 3
threshold = 100
indices = np.empty(shape=(TRIALS, cutoff), dtype=np.int)
perturbed_values = np.empty(shape=(TRIALS, cutoff), dtype=np.float)
for i in range(TRIALS):
    mechanism = SparseNumeric(epsilon=4.0, sensitivity=1.0, threshold=threshold, cutoff=cutoff)  
    indices[i], perturbed_values[i] = mechanism.release(values)

hit_names = ["Hit %i" % (j+1) for j in range(cutoff)]
value_names = ["Value %i" % (j+1) for j in range(cutoff)]
column_pairs = zip(hit_names, value_names)
column_names = [val for pair in column_pairs for val in pair]
value_pairs = zip(indices.transpose(), perturbed_values.transpose())
column_values = [val for pair in value_pairs for val in pair]
table = {k: v for (k, v) in zip(column_names, column_values)}
df = pd.DataFrame(table, index=["Trial %i" % i for i in range(TRIALS)])
display(df)
mlcat("", """Report Noisy Max
""")
mlask()

mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
from relm.mechanisms import ReportNoisyMax
mechanism = ReportNoisyMax(epsilon=1.0)  
index, perturbed_value = mechanism.release(values)
""")
mlask()
from relm.mechanisms import ReportNoisyMax
mechanism = ReportNoisyMax(epsilon=1.0)  
index, perturbed_value = mechanism.release(values)
mlcat("", """Empirical Distribution
Because ReportNoisyMax returns a single index and a single value, we can
run many experiments and plot a histogram of the results of those
experiments to get an empirical estimate of the distribution of the
mechanism’s output.
""")
mlask()

mlcat("", """
TRIALS = 2**10
indices = np.zeros(TRIALS)
perturbed_values = np.zeros(TRIALS)
for i in range(TRIALS):
    mechanism = ReportNoisyMax(epsilon=0.1)
    indices[i], perturbed_values[i] = mechanism.release(values)
""")
mlask()
TRIALS = 2**10
indices = np.zeros(TRIALS)
perturbed_values = np.zeros(TRIALS)
for i in range(TRIALS):
    mechanism = ReportNoisyMax(epsilon=0.1)
    indices[i], perturbed_values[i] = mechanism.release(values)
mlcat("", """Visualising the Results
The ReportNoisyMax mechanism resturns two values, an index and a
perturbed query response. We analyze the distribution of each output
individually.
""")
mlask()

mlcat("", """Indices
""")
mlask()

mlcat("", """
print("The index of the greatest exact count is: %i\n" % np.argmax(values))

histogram = dict.fromkeys(np.arange(min(indices), max(indices)), 0)
histogram.update(Counter(indices))

plt.xlabel("Index")
plt.ylabel("Count")
plt.title("Distribution of ReportNoisyMax output indices")
plt.plot(list(histogram.keys()), list(histogram.values()))
plt.axvline(x=np.argmax(values), color='orange')
""")
mlask()
print("The index of the greatest exact count is: %i\n" % np.argmax(values))

histogram = dict.fromkeys(np.arange(min(indices), max(indices)), 0)
histogram.update(Counter(indices))

plt.xlabel("Index")
plt.ylabel("Count")
plt.title("Distribution of ReportNoisyMax output indices")
plt.plot(list(histogram.keys()), list(histogram.values()))
plt.axvline(x=np.argmax(values), color='orange')
mlcat("", """Perturbed Values
""")
mlask()

mlcat("", """
print("The greatest exact count is: %i\n" % np.max(values))

#histogram = dict.fromkeys(np.arange(min(results), max(results)), 0)
#histogram.update(Counter(results))

plt.hist(perturbed_values, bins=128)

plt.xlabel("Perturbed Value")
plt.ylabel("Count")
plt.title("Distribution of ReportNoisyMax output values")
plt.axvline(x=np.max(values), color='orange')
""")
mlask()
print("The greatest exact count is: %i\n" % np.max(values))

#histogram = dict.fromkeys(np.arange(min(results), max(results)), 0)
#histogram.update(Counter(results))

plt.hist(perturbed_values, bins=128)

plt.xlabel("Perturbed Value")
plt.ylabel("Count")
plt.title("Distribution of ReportNoisyMax output values")
plt.axvline(x=np.max(values), color='orange')
mlcat("", """Correlated Noise Mechanisms
""")
mlask()

mlcat("", """Data Wrangling
""")
mlask()

mlcat("", """
# Generate some synthetic data
data = pd.DataFrame()
test_results = np.random.choice(["Negative", "Positive", "N/A"], size=100000, p=[0.55,0.35,0.1]) 
data["Test Result"] = test_results

# Compute a histogram representation of the data
from relm.histogram import Histogram
hist = Histogram(data)
real_database = hist.get_db()
db_size = real_database.size
db_l1_norm = real_database.sum()

# Specify the queries to be answered
queries = [[{"Test Result": "Negative"}, {"Test Result": "Positive"}],
           [{"Test Result": "Positive"}, {"Test Result": "N/A"}],
           [{"Test Result": "Negative"}, {"Test Result": "N/A"}]]
num_queries = len(queries)
queries = sps.vstack([hist.get_query_vector(q) for q in queries])

# Compute the exact query responses
values = (queries @ real_database)/real_database.sum()
""")
mlask()
# Generate some synthetic data
data = pd.DataFrame()
test_results = np.random.choice(["Negative", "Positive", "N/A"], size=100000, p=[0.55,0.35,0.1]) 
data["Test Result"] = test_results

# Compute a histogram representation of the data
from relm.histogram import Histogram
hist = Histogram(data)
real_database = hist.get_db()
db_size = real_database.size
db_l1_norm = real_database.sum()

# Specify the queries to be answered
queries = [[{"Test Result": "Negative"}, {"Test Result": "Positive"}],
           [{"Test Result": "Positive"}, {"Test Result": "N/A"}],
           [{"Test Result": "Negative"}, {"Test Result": "N/A"}]]
num_queries = len(queries)
queries = sps.vstack([hist.get_query_vector(q) for q in queries])

# Compute the exact query responses
values = (queries @ real_database)/real_database.sum()
mlcat("", """SmallDB Mechanism
""")
mlask()

mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
from relm.mechanisms import SmallDB
mechanism = SmallDB(epsilon=0.01, alpha=0.1)
synthetic_database = mechanism.release(values, queries, db_size, db_l1_norm)
""")
mlask()
from relm.mechanisms import SmallDB
mechanism = SmallDB(epsilon=0.01, alpha=0.1)
synthetic_database = mechanism.release(values, queries, db_size, db_l1_norm)
mlcat("", """Visualising the Results
""")
mlask()

mlcat("", """
TRIALS = 2**4
synthetic_responses = np.empty((TRIALS, queries.shape[0]))
for i in range(TRIALS):
    mechanism = SmallDB(epsilon=0.01, alpha=0.1)
    synthetic_database = mechanism.release(values, queries, db_size, db_l1_norm)
    synthetic_responses[i] = (queries @ synthetic_database) / synthetic_database.sum()
    
df = pd.DataFrame(data=np.row_stack((values, synthetic_responses)),
                  columns=["Query %i" % i for i in range(queries.shape[0])],
                  index=["Exact Responses"] + ["TRIAL %i" % i for i in range(TRIALS)])

display(df)
""")
mlask()
TRIALS = 2**4
synthetic_responses = np.empty((TRIALS, queries.shape[0]))
for i in range(TRIALS):
    mechanism = SmallDB(epsilon=0.01, alpha=0.1)
    synthetic_database = mechanism.release(values, queries, db_size, db_l1_norm)
    synthetic_responses[i] = (queries @ synthetic_database) / synthetic_database.sum()
    
df = pd.DataFrame(data=np.row_stack((values, synthetic_responses)),
                  columns=["Query %i" % i for i in range(queries.shape[0])],
                  index=["Exact Responses"] + ["TRIAL %i" % i for i in range(TRIALS)])

display(df)
mlcat("", """Online Multiplicative Weights Mechanism
""")
mlask()

mlcat("", """
from scipy.special import comb

# Generate some synthetic data
data = pd.DataFrame()
db_size = 8
probs = np.random.random(db_size)
probs /= probs.sum()
print(probs)
test_results = np.random.choice(np.arange(db_size), size=2**22, p=probs) 
data["Test Result"] = test_results

# Compute a histogram representation of the data
from relm.histogram import Histogram
hist = Histogram(data)
real_database = hist.get_db()
db_size = real_database.size
db_l1_norm = real_database.sum()

# Specify the queries to be answered
num_ones = 5
q_size = comb(db_size, num_ones, exact=True)
queries = [{"Test Result": k for k in np.random.choice(np.arange(db_size), size=num_ones, replace=False)} for i in range(1024)]
num_queries = len(queries)
queries = sps.vstack([hist.get_query_vector(q) for q in queries])

# Compute the exact query responses
values = (queries @ real_database)/real_database.sum()
""")
mlask()
from scipy.special import comb

# Generate some synthetic data
data = pd.DataFrame()
db_size = 8
probs = np.random.random(db_size)
probs /= probs.sum()
print(probs)
test_results = np.random.choice(np.arange(db_size), size=2**22, p=probs) 
data["Test Result"] = test_results

# Compute a histogram representation of the data
from relm.histogram import Histogram
hist = Histogram(data)
real_database = hist.get_db()
db_size = real_database.size
db_l1_norm = real_database.sum()

# Specify the queries to be answered
num_ones = 5
q_size = comb(db_size, num_ones, exact=True)
queries = [{"Test Result": k for k in np.random.choice(np.arange(db_size), size=num_ones, replace=False)} for i in range(1024)]
num_queries = len(queries)
queries = sps.vstack([hist.get_query_vector(q) for q in queries])

# Compute the exact query responses
values = (queries @ real_database)/real_database.sum()
mlcat("", """Basic Usage
""")
mlask()

mlcat("", """
from relm.mechanisms import PrivateMultiplicativeWeights
mechanism = PrivateMultiplicativeWeights(epsilon=1.0, alpha=0.15, beta=0.01, q_size=q_size, db_size=db_size, db_l1_norm=db_l1_norm)
dp_responses = mechanism.release(values, queries)
""")
mlask()
from relm.mechanisms import PrivateMultiplicativeWeights
mechanism = PrivateMultiplicativeWeights(epsilon=1.0, alpha=0.15, beta=0.01, q_size=q_size, db_size=db_size, db_l1_norm=db_l1_norm)
dp_responses = mechanism.release(values, queries)
mlcat("", """
print(values)
print(dp_responses)
""")
mlask()
print(values)
print(dp_responses)
mlcat("", """
print(mechanism.db_l1_norm)
print(mechanism.cutoff)
print(mechanism.q_size)
print(mechanism.beta)
print(mechanism.est_data)
print(mechanism.threshold)
""")
mlask()
print(mechanism.db_l1_norm)
print(mechanism.cutoff)
print(mechanism.q_size)
print(mechanism.beta)
print(mechanism.est_data)
print(mechanism.threshold)
mlcat("", """
print(mechanism.alpha)
temp = 36*np.log(mechanism.db_size)
temp *= np.log(mechanism.q_size) + np.log(32*mechanism.db_size/(mechanism.alpha**2 * mechanism.beta))
temp /= mechanism.epsilon * mechanism.db_l1_norm * mechanism.alpha**2
print(temp)
""")
mlask()
print(mechanism.alpha)
temp = 36*np.log(mechanism.db_size)
temp *= np.log(mechanism.q_size) + np.log(32*mechanism.db_size/(mechanism.alpha**2 * mechanism.beta))
temp /= mechanism.epsilon * mechanism.db_l1_norm * mechanism.alpha**2
print(temp)
mlcat("", """
np.max(np.abs(values - dp_responses))
""")
mlask()
np.max(np.abs(values - dp_responses))
mlcat("", """

""")
mlask()

mlcat("", """

""")
mlask()

mlcat("", """

""")
mlask()

