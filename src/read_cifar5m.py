dog = np.load('/n/holystore01/LABS/barak_lab/Everyone/cifar-5m/class5.npy')

dog_train = dog[(train_sz // 2) * (offset):(train_sz // 2) * (1 + offset)]
dog_train2 = dog[260000 + (train2_sz // 2) * (offset):260000 + (train2_sz // 2) * (1 + offset)]
dog_test = dog[-test_sz // 2:]

horse = np.load('/n/holystore01/LABS/barak_lab/Everyone/cifar-5m/class7.npy')
horse_train = horse[(train_sz // 2) * (offset):(train_sz // 2) * (1 + offset)]
horse_train2 = horse[260000 + (train2_sz // 2) * (offset):260000 + (train2_sz // 2) * (1 + offset)]
horse_test = horse[-test_sz // 2:]

mean = np.array([0.4555, 0.4362, 0.3415])[None, None, None, :]
std = np.array([.2284, .2167, .2165])[None, None, None, :]

x_train = np.array((np.concatenate([dog_train, horse_train]) / 255. - mean) / std)
x_train2 = np.array((np.concatenate([dog_train2, horse_train2]) / 255. - mean) / std)
x_test = np.array((np.concatenate([dog_test, horse_test]) / 255. - mean) / std)

y_train = np.concatenate((np.zeros((train_sz // 2)) + 1., np.zeros((train_sz // 2)) - 1.)).reshape((train_sz, 1))
y_train2 = np.concatenate((np.zeros((train2_sz // 2)) + 1., np.zeros((train2_sz // 2)) - 1.)).reshape((train2_sz, 1))
y_test = np.concatenate((np.zeros((test_sz // 2)) + 1., np.zeros((test_sz // 2)) - 1.)).reshape((test_sz, 1))

print(x_train.shape, y_train.shape)
print(x_train2.shape, y_train2.shape)
print(x_test.shape, y_test.shape)

train_dataset = C10_Dataset(x_train, y_train)
train2_dataset = C10_Dataset(x_train2, y_train2)
test_dataset = C10_Dataset(x_test, y_test)

rng = jax.random.PRNGKey(0)

train_dataloader = JaxDataLoader(train_dataset, rng, batch_size=batch_size, shuffle=shuffle, collate_fn=stack_collate,
                                 num_workers=2)
train2_dataloader = JaxDataLoader(train2_dataset, rng, batch_size=batch_size, shuffle=shuffle, collate_fn=stack_collate,
                                  num_workers=2)
test_dataloader = JaxDataLoader(test_dataset, rng, batch_size=batch_test_size, shuffle=False, collate_fn=stack_collate,
                                num_workers=2)

dog = np.load('/n/holystore01/LABS/barak_lab/Everyone/cifar-5m/class5.npy')

dog_train = dog[(train_sz // 2) * (offset):(train_sz // 2) * (1 + offset)]
dog_train2 = dog[260000 + (train2_sz // 2) * (offset):260000 + (train2_sz // 2) * (1 + offset)]
dog_test = dog[-test_sz // 2:]

horse = np.load('/n/holystore01/LABS/barak_lab/Everyone/cifar-5m/class7.npy')
horse_train = horse[(train_sz // 2) * (offset):(train_sz // 2) * (1 + offset)]
horse_train2 = horse[260000 + (train2_sz // 2) * (offset):260000 + (train2_sz // 2) * (1 + offset)]
horse_test = horse[-test_sz // 2:]

mean = np.array([0.4555, 0.4362, 0.3415])[None, None, None, :]
std = np.array([.2284, .2167, .2165])[None, None, None, :]

x_train = np.array((np.concatenate([dog_train, horse_train]) / 255. - mean) / std)
x_train2 = np.array((np.concatenate([dog_train2, horse_train2]) / 255. - mean) / std)
x_test = np.array((np.concatenate([dog_test, horse_test]) / 255. - mean) / std)

y_train = np.concatenate((np.zeros((train_sz // 2)) + 1., np.zeros((train_sz // 2)) - 1.)).reshape((train_sz, 1))
y_train2 = np.concatenate((np.zeros((train2_sz // 2)) + 1., np.zeros((train2_sz // 2)) - 1.)).reshape((train2_sz, 1))
y_test = np.concatenate((np.zeros((test_sz // 2)) + 1., np.zeros((test_sz // 2)) - 1.)).reshape((test_sz, 1))

print(x_train.shape, y_train.shape)
print(x_train2.shape, y_train2.shape)
print(x_test.shape, y_test.shape)

train_dataset = C10_Dataset(x_train, y_train)
train2_dataset = C10_Dataset(x_train2, y_train2)
test_dataset = C10_Dataset(x_test, y_test)

rng = jax.random.PRNGKey(0)

train_dataloader = JaxDataLoader(train_dataset, rng, batch_size=batch_size, shuffle=shuffle, collate_fn=stack_collate,
                                 num_workers=2)
train2_dataloader = JaxDataLoader(train2_dataset, rng, batch_size=batch_size, shuffle=shuffle, collate_fn=stack_collate,
                                  num_workers=2)
test_dataloader = JaxDataLoader(test_dataset, rng, batch_size=batch_test_size, shuffle=False, collate_fn=stack_collate,
                                num_workers=2)