indices = find(tr.trainMask{1} == 1);

TRAIN_INPUT = train_input(:, indices);
writematrix(TRAIN_INPUT, "personal_train_input.txt")

TRAIN_TARGET = train_target(:, indices);
writematrix(TRAIN_TARGET, "personal_train_target.txt")

TEST_INPUT = test_input;
writematrix(TEST_INPUT, "personal_test_input.txt")

TEST_TARGET = test_target;
writematrix(TEST_TARGET, "personal_test_target.txt")

PREDS = preds;
writematrix(PREDS, "personal_test_preds.txt")