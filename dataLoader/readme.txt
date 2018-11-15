
# 所有的数据加载在这里完成
# 通过调用dataLoader/__init__.py的函数返回torch.utils.data.DataLoader类型
# 注意加载数据时候，sampler:

# create sampler
if train_opts.sampler == 'stratified':
    print('stratified sampler')
    train_sampler = StratifiedSampler(train_dataset.labels, train_opts.batchSize)
    batch_size = 52
elif train_opts.sampler == 'weighted2':
    print('weighted sampler with background weight={}x'.format(train_opts.bgd_weight_multiplier))
    # modify and increase background weight
    weight = train_dataset.weight
    bgd_weight = np.min(weight)
    weight[abs(weight - bgd_weight) < 1e-8] = bgd_weight * train_opts.bgd_weight_multiplier
    train_sampler = sampler.WeightedRandomSampler(weight, len(train_dataset.weight))
    batch_size = train_opts.batchSize
else:
    print('weighted sampler')
    train_sampler = sampler.WeightedRandomSampler(train_dataset.weight, len(train_dataset.weight))
    batch_size = train_opts.batchSize

train_loader = DataLoader(dataset=train_dataset, num_workers=num_workers,
                          batch_size=batch_size, sampler=train_sampler)


