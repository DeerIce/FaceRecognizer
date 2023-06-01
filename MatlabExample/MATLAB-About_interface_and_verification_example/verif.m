function verif()
    load same_pairs;
    load diff_pairs;
    result_same = [];
    result_diff = [];
    num_same = length(same_pairs);
    num_diff = length(diff_pairs);
    for i = 1:num_same
        img_path1 = same_pairs{i, 1};
        img_path2 = same_pairs{i, 2};
        fprintf('same-%d  %s  %s\n', i, img_path1, img_path2);
        result_same = [result_same; FaceVerification(img_path1, img_path2)];
    end
    for i = 1:num_diff
        img_path1 = diff_pairs{i, 1};
        img_path2 = diff_pairs{i, 2};
        fprintf('diff-%d  %s  %s\n', i, img_path1, img_path2);
        result_diff = [result_diff; FaceVerification(img_path1, img_path2)];
    end
    num_right_same = num_same - sum(xor(result_same,  ones(num_same, 1)));
    num_right_diff = num_diff - sum(xor(result_diff, zeros(num_same, 1)));
    acc = (num_right_diff+num_right_same) / (num_diff+num_same);
    fprintf('num_right_same = %d(%.4f), num_right_diff = %d(%.4f). accuracy = %.4f\n', num_right_same, num_right_same/num_same, num_right_diff, num_right_diff/num_diff, acc);
end
