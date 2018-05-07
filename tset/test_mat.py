import scipy.io as scio

label_file = '/home/yangshun/.cache/paddle/dataset/flowers/imagelabels.mat'
setid_file = '/home/yangshun/.cache/paddle/dataset/flowers/setid.mat'

labels = scio.loadmat(label_file)['labels']
indexes = scio.loadmat(setid_file)
print(len(indexes['tstid'][0]))
print(len(indexes['trnid'][0]))
print(len(indexes['valid'][0]))
print(len(labels[0]))
