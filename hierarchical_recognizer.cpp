/*
 * hierarchical_recognizer.cpp
 * 階層構造的画像パターン認識器 - C++実装（計算コア）
 * 
 * コンパイル:
 *   g++ -shared -fPIC -O3 -o hierarchical_recognizer.so hierarchical_recognizer.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

#define LINE_MAX 80

extern "C" {

//==============================================================
// データ構造定義
//==============================================================

// RBFユニット（カーネルメモリーの中間層）
typedef struct {
    int vec_len;           // ベクトル長
    double *centroid;      // セントロイドベクトル
    double activation;     // 活性化値 h
    double dist_sq;        // 距離の2乗
} RBFUnit;

// サブネット（1クラスに対応）
typedef struct {
    int class_id;          // クラスID
    int num_units;         // RBFユニット数
    int vec_len;           // ベクトル長
    RBFUnit *units;        // RBFユニット配列
    double output;         // サブネット出力
} SubNet;

// カーネルメモリー分類器
typedef struct {
    int num_subnets;       // サブネット数
    SubNet *subnets;       // サブネット配列
    double radius_denom;   // 半径値の分母
} KernelMemory;

// 階層ピラミッド構造
typedef struct {
    int num_levels;        // 階層数
    int *widths;           // 各階層の幅
    int *heights;          // 各階層の高さ
    double **layers;       // 各階層の画像データ（正規化済み）
} Pyramid;

// 階層構造的認識器（論文3.6節の構造）
typedef struct {
    KernelMemory *common_classifier;      // 共通部分類器
    KernelMemory **non_common_classifiers; // 非共通部分類器配列
    int num_common_groups;                 // 共通部グループ数
    int *common_group_ids;                 // 共通部グループID配列
    int num_pyramid_levels;                // ピラミッド階層数
    int common_layer_index;                // 共通部で使用する階層
    int non_common_layer_index;            // 非共通部で使用する階層
} HierarchicalRecognizer;

//==============================================================
// ユーティリティ関数
//==============================================================

void show_progress(int current, int total, const char* task_name) {
    if (current % 100 == 0 || current == total - 1) {
        printf("  %s: %d/%d\n", task_name, current + 1, total);
    }
}

//==============================================================
// RBFユニット操作
//==============================================================

void freeRBFUnit(RBFUnit *unit) {
    if (unit && unit->centroid) {
        free(unit->centroid);
    }
}

//==============================================================
// サブネット操作
//==============================================================

void addRBFToSubNet(SubNet *subnet, const double *centroid) {
    int n = subnet->num_units;
    subnet->units = (RBFUnit*)realloc(subnet->units, (n + 1) * sizeof(RBFUnit));
    
    subnet->units[n].vec_len = subnet->vec_len;
    subnet->units[n].centroid = (double*)calloc(subnet->vec_len, sizeof(double));
    memcpy(subnet->units[n].centroid, centroid, subnet->vec_len * sizeof(double));
    subnet->units[n].activation = 0.0;
    subnet->units[n].dist_sq = 0.0;
    
    subnet->num_units++;
}

void freeSubNet(SubNet *subnet) {
    if (subnet) {
        for (int i = 0; i < subnet->num_units; i++) {
            freeRBFUnit(&subnet->units[i]);
        }
        if (subnet->units) free(subnet->units);
    }
}

//==============================================================
// カーネルメモリー分類器操作（論文3.3節）
//==============================================================

API KernelMemory* createKernelMemory() {
    KernelMemory *km = (KernelMemory*)calloc(1, sizeof(KernelMemory));
    km->num_subnets = 0;
    km->subnets = NULL;
    km->radius_denom = 1.0;
    return km;
}

void addSubNetToKM(KernelMemory *km, int class_id, const double *centroid, int vec_len) {
    int n = km->num_subnets;
    km->subnets = (SubNet*)realloc(km->subnets, (n + 1) * sizeof(SubNet));
    
    km->subnets[n].class_id = class_id;
    km->subnets[n].num_units = 0;
    km->subnets[n].vec_len = vec_len;
    km->subnets[n].units = NULL;
    km->subnets[n].output = 0.0;
    
    addRBFToSubNet(&km->subnets[n], centroid);
    km->num_subnets++;
}

int getMaxActivatedRBF(const SubNet *subnet) {
    int max_idx = 0;
    double max_act = subnet->units[0].activation;
    
    for (int i = 1; i < subnet->num_units; i++) {
        if (subnet->units[i].activation > max_act) {
            max_act = subnet->units[i].activation;
            max_idx = i;
        }
    }
    return max_idx;
}

// 論文3.3節: カーネルメモリーの順伝播
int forwardKernelMemory(KernelMemory *km, const double *x, int x_len) {
    // Step 1: 全RBFのL2距離の2乗を計算し、最大値を求める
    double max_dist_sq = -1.0;
    
    for (int i = 0; i < km->num_subnets; i++) {
        SubNet *subnet = &km->subnets[i];
        if (subnet->vec_len != x_len) continue;
        
        for (int j = 0; j < subnet->num_units; j++) {
            RBFUnit *unit = &subnet->units[j];
            double dist_sq = 0.0;
            
            for (int k = 0; k < x_len; k++) {
                double diff = x[k] - unit->centroid[k];
                dist_sq += diff * diff;
            }
            
            unit->dist_sq = dist_sq;
            if (max_dist_sq < dist_sq) {
                max_dist_sq = dist_sq;
            }
        }
    }
    
    // Step 2: RBF出力を計算し、サブネット出力を求める
    // σ² = max_dist_sq / r_den²
    double sigma_sq = (max_dist_sq > 0) ? 
                      (max_dist_sq / (km->radius_denom * km->radius_denom)) : 1.0;
    
    int max_subnet_idx = -1;
    double max_output = -1.0;
    
    for (int i = 0; i < km->num_subnets; i++) {
        SubNet *subnet = &km->subnets[i];
        if (subnet->vec_len != x_len) continue;
        
        double sum = 0.0;
        for (int j = 0; j < subnet->num_units; j++) {
            RBFUnit *unit = &subnet->units[j];
            // h_k(j) = exp(-||x - w_k(j)||² / σ²)
            unit->activation = exp(-unit->dist_sq / sigma_sq);
            sum += unit->activation;
        }
        
        // y_k = Σh_k(j) / M_k
        subnet->output = sum / subnet->num_units;
        
        if (max_output < subnet->output) {
            max_output = subnet->output;
            max_subnet_idx = i;
        }
    }
    
    return max_subnet_idx;
}

// 論文3.4節: カーネルメモリーの学習アルゴリズム
void trainKernelMemory(KernelMemory *km, double **x_train, int *y_train, int n_train,
                       int vec_len, int total_classes, int class_num_method,
                       int radius_method, int weight_update_method, double custom_radius) {
    
    if (n_train == 0) return;
    
    // Step 1: 初期設定
    addSubNetToKM(km, y_train[0], x_train[0], vec_len);
    
    // 半径分母の設定
    if (class_num_method == 0) { // INCREMENTAL
        if (radius_method == 0) {
            km->radius_denom = 1.0;
        } else if (radius_method == 1) {
            km->radius_denom = (double)total_classes;
        } else {
            km->radius_denom = custom_radius;
        }
    } else { // FIXED
        km->radius_denom = (double)total_classes;
    }
    
    // Step 2: 各学習サンプルについて処理
    for (int i = 1; i < n_train; i++) {
        if (i % 1000 == 0) {
            printf("    学習進捗: %d/%d (サブネット数: %d)\n", i, n_train, km->num_subnets);
        }
        
        int target_class = y_train[i];
        
        // サブネットの存在確認
        int subnet_idx = -1;
        for (int j = 0; j < km->num_subnets; j++) {
            if (km->subnets[j].class_id == target_class) {
                subnet_idx = j;
                break;
            }
        }
        
        if (subnet_idx == -1) {
            // 新規サブネット追加
            addSubNetToKM(km, target_class, x_train[i], vec_len);
            
            // 半径分母更新（INCREMENTAL の場合）
            if (class_num_method == 0 && radius_method == 0) {
                km->radius_denom = (double)km->num_subnets;
            }
        } else {
            // 順伝播
            int max_subnet_idx = forwardKernelMemory(km, x_train[i], vec_len);
            SubNet *max_subnet = &km->subnets[max_subnet_idx];
            
            if (max_subnet->class_id != target_class) {
                // 誤分類: 正しいサブネットにRBF追加
                addRBFToSubNet(&km->subnets[subnet_idx], x_train[i]);
            } else {
                // 正分類: 結合係数更新（オプション）
                if (weight_update_method == 1) { // AVERAGE
                    int max_rbf_idx = getMaxActivatedRBF(max_subnet);
                    RBFUnit *rbf = &max_subnet->units[max_rbf_idx];
                    
                    for (int k = 0; k < vec_len; k++) {
                        rbf->centroid[k] = (rbf->centroid[k] + x_train[i][k]) / 2.0;
                    }
                }
            }
        }
    }
}

API void freeKernelMemory(KernelMemory *km) {
    if (km) {
        for (int i = 0; i < km->num_subnets; i++) {
            freeSubNet(&km->subnets[i]);
        }
        if (km->subnets) free(km->subnets);
        free(km);
    }
}

//==============================================================
// 階層ピラミッド構造操作（論文3.1節 Step 1）
//==============================================================

// 画像のダウンサンプリング（量子化）
double* quantizeImage(const double *img, int width, int height, 
                     int *new_width, int *new_height) {
    *new_width = width / 2;
    *new_height = height / 2;
    
    if (*new_width < 1) *new_width = 1;
    if (*new_height < 1) *new_height = 1;
    
    double *quantized = (double*)calloc(*new_width * *new_height, sizeof(double));
    
    for (int y = 0; y < *new_height; y++) {
        for (int x = 0; x < *new_width; x++) {
            double sum = 0.0;
            int count = 0;
            
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int sy = y * 2 + dy;
                    int sx = x * 2 + dx;
                    if (sy < height && sx < width) {
                        sum += img[sy * width + sx];
                        count++;
                    }
                }
            }
            
            quantized[y * (*new_width) + x] = sum / count;
        }
    }
    
    return quantized;
}

API Pyramid* createPyramid(const double *img, int width, int height, int num_levels) {
    Pyramid *pyramid = (Pyramid*)calloc(1, sizeof(Pyramid));
    pyramid->num_levels = num_levels;
    pyramid->widths = (int*)calloc(num_levels, sizeof(int));
    pyramid->heights = (int*)calloc(num_levels, sizeof(int));
    pyramid->layers = (double**)calloc(num_levels, sizeof(double*));
    
    // 最下層（インデックス num_levels-1）は元画像
    pyramid->widths[num_levels - 1] = width;
    pyramid->heights[num_levels - 1] = height;
    pyramid->layers[num_levels - 1] = (double*)calloc(width * height, sizeof(double));
    memcpy(pyramid->layers[num_levels - 1], img, width * height * sizeof(double));
    
    // 下から上へ量子化
    const double *current = img;
    int cur_width = width;
    int cur_height = height;
    
    for (int level = num_levels - 2; level >= 0; level--) {
        int new_width, new_height;
        double *quantized = quantizeImage(current, cur_width, cur_height, 
                                         &new_width, &new_height);
        
        pyramid->widths[level] = new_width;
        pyramid->heights[level] = new_height;
        pyramid->layers[level] = quantized;
        
        if (level > 0) {
            double *temp = (double*)calloc(new_width * new_height, sizeof(double));
            memcpy(temp, quantized, new_width * new_height * sizeof(double));
            if (level < num_levels - 2 && current != img) {
                free((void*)current);
            }
            current = temp;
            cur_width = new_width;
            cur_height = new_height;
        }
    }
    
    if (current != img && num_levels > 1) {
        free((void*)current);
    }
    
    return pyramid;
}

API void freePyramid(Pyramid *pyramid) {
    if (pyramid) {
        for (int i = 0; i < pyramid->num_levels; i++) {
            if (pyramid->layers[i]) free(pyramid->layers[i]);
        }
        if (pyramid->widths) free(pyramid->widths);
        if (pyramid->heights) free(pyramid->heights);
        if (pyramid->layers) free(pyramid->layers);
        free(pyramid);
    }
}

//==============================================================
// 階層構造的認識器の構築（論文3.1節～3.6節）
//==============================================================

API HierarchicalRecognizer* createHierarchicalRecognizer(int num_pyramid_levels) {
    HierarchicalRecognizer *hr = (HierarchicalRecognizer*)calloc(1, sizeof(HierarchicalRecognizer));
    hr->common_classifier = createKernelMemory();
    hr->non_common_classifiers = NULL;
    hr->num_common_groups = 0;
    hr->common_group_ids = NULL;
    hr->num_pyramid_levels = num_pyramid_levels;
    hr->common_layer_index = num_pyramid_levels / 2;  // 中間層
    hr->non_common_layer_index = num_pyramid_levels - 1;  // 最下層
    return hr;
}

// 論文3.1節 Step 2: 学習
API void trainHierarchicalRecognizer(
    HierarchicalRecognizer *hr,
    double **images,           // 前処理済み画像（1次元化）
    int *image_widths,         // 各画像の幅
    int *image_heights,        // 各画像の高さ
    int *class_labels,         // クラスラベル
    int *common_labels,        // 共通部ラベル
    int n_samples,             // サンプル数
    int total_classes,         // 総クラス数
    int class_num_method,      // クラス数設定法
    int radius_method,         // 半径値設定法
    int weight_update_method,  // 結合係数更新法
    double custom_radius       // カスタム半径値
) {
    printf("階層構造的認識器の学習開始\n");
    printf("サンプル数: %d, クラス数: %d\n", n_samples, total_classes);
    
    // Step 1: 階層ピラミッド構造の生成
    printf("\nStep 1: 階層ピラミッド構造の生成\n");
    Pyramid **pyramids = (Pyramid**)calloc(n_samples, sizeof(Pyramid*));
    
    for (int i = 0; i < n_samples; i++) {
        show_progress(i, n_samples, "ピラミッド生成");
        pyramids[i] = createPyramid(images[i], image_widths[i], image_heights[i], 
                                    hr->num_pyramid_levels);
    }
    
    // 共通部グループの抽出
    int max_common = -1;
    for (int i = 0; i < n_samples; i++) {
        if (common_labels[i] > max_common) {
            max_common = common_labels[i];
        }
    }
    hr->num_common_groups = max_common + 1;
    hr->common_group_ids = (int*)calloc(hr->num_common_groups, sizeof(int));
    for (int i = 0; i < hr->num_common_groups; i++) {
        hr->common_group_ids[i] = i;
    }
    
    // Step 2: 共通部パターン分類器の構築（論文3.1節 Step 2.3）
    printf("\nStep 2: 共通部パターン分類器の構築\n");
    
    // 共通部の特徴を抽出
    double **common_features = (double**)calloc(n_samples, sizeof(double*));
    int *common_feature_lens = (int*)calloc(n_samples, sizeof(int));
    
    for (int i = 0; i < n_samples; i++) {
        Pyramid *p = pyramids[i];
        int layer = hr->common_layer_index;
        int size = p->widths[layer] * p->heights[layer];
        
        common_features[i] = (double*)calloc(size, sizeof(double));
        memcpy(common_features[i], p->layers[layer], size * sizeof(double));
        common_feature_lens[i] = size;
    }
    
    // 共通部分類器の学習
    trainKernelMemory(hr->common_classifier, common_features, common_labels, n_samples,
                     common_feature_lens[0], hr->num_common_groups,
                     class_num_method, radius_method, weight_update_method, custom_radius);
    
    // Step 3: 非共通部パターン分類器の構築（論文3.1節 Step 2.4）
    printf("\nStep 3: 非共通部パターン分類器の構築\n");
    
    hr->non_common_classifiers = (KernelMemory**)calloc(hr->num_common_groups, 
                                                         sizeof(KernelMemory*));
    
    for (int common_id = 0; common_id < hr->num_common_groups; common_id++) {
        printf("  共通部グループ %d の処理\n", common_id);
        
        // このグループに属するサンプルを抽出
        int group_count = 0;
        for (int i = 0; i < n_samples; i++) {
            if (common_labels[i] == common_id) group_count++;
        }
        
        if (group_count == 0) continue;
        
        double **group_features = (double**)calloc(group_count, sizeof(double*));
        int *group_labels = (int*)calloc(group_count, sizeof(int));
        int group_idx = 0;
        
        for (int i = 0; i < n_samples; i++) {
            if (common_labels[i] == common_id) {
                Pyramid *p = pyramids[i];
                int layer = hr->non_common_layer_index;
                int size = p->widths[layer] * p->heights[layer];
                
                group_features[group_idx] = (double*)calloc(size, sizeof(double));
                memcpy(group_features[group_idx], p->layers[layer], size * sizeof(double));
                group_labels[group_idx] = class_labels[i];
                group_idx++;
            }
        }
        
        // 非共通部分類器の学習
        hr->non_common_classifiers[common_id] = createKernelMemory();
        
        int vec_len = pyramids[0]->widths[hr->non_common_layer_index] * 
                      pyramids[0]->heights[hr->non_common_layer_index];
        
        trainKernelMemory(hr->non_common_classifiers[common_id], 
                         group_features, group_labels, group_count,
                         vec_len, total_classes,
                         class_num_method, radius_method, weight_update_method, custom_radius);
        
        // メモリ解放
        for (int i = 0; i < group_count; i++) {
            free(group_features[i]);
        }
        free(group_features);
        free(group_labels);
    }
    
    // クリーンアップ
    for (int i = 0; i < n_samples; i++) {
        freePyramid(pyramids[i]);
        free(common_features[i]);
    }
    free(pyramids);
    free(common_features);
    free(common_feature_lens);
    
    printf("\n学習完了\n");
}

// 論文3.6節: 認識フェーズ
API int predictHierarchicalRecognizer(
    HierarchicalRecognizer *hr,
    const double *image,       // 前処理済み画像
    int width,
    int height
) {
    // Step 1: 階層ピラミッド構造の生成
    Pyramid *pyramid = createPyramid(image, width, height, hr->num_pyramid_levels);
    
    // Step 2: 共通部検出・分類
    int common_layer = hr->common_layer_index;
    int common_size = pyramid->widths[common_layer] * pyramid->heights[common_layer];
    double *common_feature = pyramid->layers[common_layer];
    
    int common_subnet_idx = forwardKernelMemory(hr->common_classifier, 
                                                common_feature, common_size);
    int common_id = hr->common_classifier->subnets[common_subnet_idx].class_id;
    
    // Step 3: 非共通部分類
    int final_class_id = common_id;  // デフォルト
    
    if (common_id >= 0 && common_id < hr->num_common_groups && 
        hr->non_common_classifiers[common_id] != NULL) {
        
        int non_common_layer = hr->non_common_layer_index;
        int non_common_size = pyramid->widths[non_common_layer] * 
                             pyramid->heights[non_common_layer];
        double *non_common_feature = pyramid->layers[non_common_layer];
        
        int non_common_subnet_idx = forwardKernelMemory(
            hr->non_common_classifiers[common_id],
            non_common_feature, non_common_size
        );
        
        final_class_id = hr->non_common_classifiers[common_id]->subnets[non_common_subnet_idx].class_id;
    }
    
    freePyramid(pyramid);
    return final_class_id;
}

API void freeHierarchicalRecognizer(HierarchicalRecognizer *hr) {
    if (hr) {
        if (hr->common_classifier) freeKernelMemory(hr->common_classifier);
        
        if (hr->non_common_classifiers) {
            for (int i = 0; i < hr->num_common_groups; i++) {
                if (hr->non_common_classifiers[i]) {
                    freeKernelMemory(hr->non_common_classifiers[i]);
                }
            }
            free(hr->non_common_classifiers);
        }
        
        if (hr->common_group_ids) free(hr->common_group_ids);
        free(hr);
    }
}

//==============================================================
// バッチ予測
//==============================================================

API int* batchPredictHierarchical(
    HierarchicalRecognizer *hr,
    double **images,
    int *widths,
    int *heights,
    int n_samples
) {
    int *predictions = (int*)calloc(n_samples, sizeof(int));
    
    printf("バッチ予測開始: %dサンプル\n", n_samples);
    for (int i = 0; i < n_samples; i++) {
        show_progress(i, n_samples, "予測");
        predictions[i] = predictHierarchicalRecognizer(hr, images[i], widths[i], heights[i]);
    }
    printf("\n");
    
    return predictions;
}

API void freeIntArray(int *arr) {
    if (arr) free(arr);
}

//==============================================================
// 統計情報取得
//==============================================================

// RBF統計情報の構造体
typedef struct {
    int total_rbf_count;           // 総RBF数
    int common_rbf_count;          // 共通部RBF数
    int *non_common_rbf_counts;    // 各非共通部のRBF数
    int num_groups;                // グループ数
    int *class_rbf_counts;         // クラスごとのRBF数
    int num_classes;               // クラス数
} RBFStatistics;

API RBFStatistics* getRBFStatistics(HierarchicalRecognizer *hr) {
    RBFStatistics *stats = (RBFStatistics*)calloc(1, sizeof(RBFStatistics));

    // 共通部のRBF数を集計
    stats->common_rbf_count = 0;
    if (hr->common_classifier) {
        for (int i = 0; i < hr->common_classifier->num_subnets; i++) {
            stats->common_rbf_count += hr->common_classifier->subnets[i].num_units;
        }
    }

    // 非共通部のRBF数を集計
    stats->num_groups = hr->num_common_groups;
    stats->non_common_rbf_counts = (int*)calloc(stats->num_groups, sizeof(int));

    for (int i = 0; i < hr->num_common_groups; i++) {
        stats->non_common_rbf_counts[i] = 0;
        if (hr->non_common_classifiers[i]) {
            for (int j = 0; j < hr->non_common_classifiers[i]->num_subnets; j++) {
                stats->non_common_rbf_counts[i] +=
                    hr->non_common_classifiers[i]->subnets[j].num_units;
            }
        }
    }

    // 総RBF数
    stats->total_rbf_count = stats->common_rbf_count;
    for (int i = 0; i < stats->num_groups; i++) {
        stats->total_rbf_count += stats->non_common_rbf_counts[i];
    }

    // クラスごとのRBF数を集計
    // 最大クラスIDを見つける
    int max_class_id = -1;
    for (int i = 0; i < hr->num_common_groups; i++) {
        if (hr->non_common_classifiers[i]) {
            for (int j = 0; j < hr->non_common_classifiers[i]->num_subnets; j++) {
                int class_id = hr->non_common_classifiers[i]->subnets[j].class_id;
                if (class_id > max_class_id) max_class_id = class_id;
            }
        }
    }

    stats->num_classes = max_class_id + 1;
    stats->class_rbf_counts = (int*)calloc(stats->num_classes, sizeof(int));

    // クラスごとにRBF数をカウント
    for (int i = 0; i < hr->num_common_groups; i++) {
        if (hr->non_common_classifiers[i]) {
            for (int j = 0; j < hr->non_common_classifiers[i]->num_subnets; j++) {
                int class_id = hr->non_common_classifiers[i]->subnets[j].class_id;
                stats->class_rbf_counts[class_id] =
                    hr->non_common_classifiers[i]->subnets[j].num_units;
            }
        }
    }

    return stats;
}

API void freeRBFStatistics(RBFStatistics *stats) {
    if (stats) {
        if (stats->non_common_rbf_counts) free(stats->non_common_rbf_counts);
        if (stats->class_rbf_counts) free(stats->class_rbf_counts);
        free(stats);
    }
}

} // extern "C"