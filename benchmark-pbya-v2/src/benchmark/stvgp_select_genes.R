# Replicate stVGP's R-based Moran's I gene selection
# (from Tutorial/Spatial genetic analysis BC.R)
#
# Usage: Rscript stvgp_select_genes.R <input_dir> <output_dir> <n_slices> <spot_make>
# Input: <input_dir>/select_gene_<spot_make^2>.txt per slice (from Python select_gene)
# Output: <output_dir>/<slice_idx>/gene_morans_<spot_make^2>.txt

args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[1]
output_dir <- args[2]
slice_num <- as.integer(args[3]) - 1  # 0-indexed
spot_make <- as.integer(args[4])
n_quadrants <- spot_make * spot_make

library(Seurat)
library(sp)
library(spdep)

for (index_slice in 0:slice_num) {
  input_file <- file.path(input_dir, index_slice, paste0("select_gene_", n_quadrants, ".txt"))
  if (!file.exists(input_file)) {
    cat("SKIP: missing", input_file, "\n")
    next
  }

  spotmarker_data <- read.table(input_file, header = TRUE, check.names = FALSE)

  cluster_class <- spotmarker_data["marker_cluster", ]
  cluster_class <- t(cluster_class)
  cluster_class <- as.data.frame(cluster_class)

  spatial_coordinates <- spotmarker_data[c("x", "y"), ]
  spatial_coordinates <- t(spatial_coordinates)
  spatial_coordinates <- as.data.frame(spatial_coordinates)

  spotmarker_data <- spotmarker_data[!(rownames(spotmarker_data) == "marker_cluster"), ]
  spotmarker_data <- spotmarker_data[!(rownames(spotmarker_data) == "x"), ]
  spotmarker_data <- spotmarker_data[!(rownames(spotmarker_data) == "y"), ]

  new_class_vector <- as.vector(cluster_class)
  my_vector <- unlist(new_class_vector)
  levels <- unique(my_vector)
  levels <- as.character(levels)
  my_vector22 <- factor(my_vector, levels = levels)

  # Data is already log-normalized from Python preprocessing.
  seurat_obj <- CreateSeuratObject(counts = spotmarker_data)
  seurat_obj <- SetAssayData(seurat_obj, layer = "data",
                             new.data = as.matrix(GetAssayData(seurat_obj, layer = "counts")))
  # Fix: names must match Seurat cell names for Idents to work
  names(my_vector22) <- colnames(seurat_obj)
  Idents(seurat_obj) <- my_vector22

  seurat_obj <- FindVariableFeatures(seurat_obj)
  # Use low thresholds since data is log-normalized (not raw counts)
  all_markers <- FindAllMarkers(seurat_obj, only.pos = TRUE,
                                logfc.threshold = 0.01, min.pct = 0.01,
                                test.use = "wilcox", slot = "data")

  gene_name <- unique(all_markers$gene)
  cat("Slice", index_slice, ":", length(gene_name), "marker genes\n")

  spotmarker_data <- t(spotmarker_data)
  spotmarker_data <- as.data.frame(spotmarker_data)
  spotmarker_data$x <- spatial_coordinates$x
  spotmarker_data$y <- spatial_coordinates$y
  spotmarker_data$marker_cluster <- cluster_class$marker_cluster

  morans_file <- spotmarker_data
  cluster_index <- unique(morans_file$marker_cluster)
  index_row_col <- trunc(sqrt(max(cluster_index)))

  result_gene_morans <- data.frame(stringsAsFactors = FALSE)

  for (gene in gene_name) {
    result_matrix <- matrix(NA, nrow = index_row_col, ncol = index_row_col)
    for (idx in cluster_index) {
      subset_morans_file <- morans_file[morans_file$marker_cluster == idx, ]
      coordinates <- subset_morans_file[, c("x", "y")]

      if (nrow(coordinates) < 7) {
        result_matrix[idx] <- 0
        next
      }

      neighbors <- knn2nb(knearneigh(coordinates, k = 6))
      W <- nb2listw(neighbors)

      if (all(subset_morans_file[[gene]] == 0)) {
        result_matrix[idx] <- 0
        next
      }

      moran_result <- moran.mc(subset_morans_file[[gene]], W, nsim = 999, zero.policy = TRUE)
      result_matrix[idx] <- moran_result$statistic
    }
    new_matrix <- result_matrix - mean(result_matrix, na.rm = TRUE)
    new_matrix <- new_matrix * new_matrix
    var_value <- sum(new_matrix, na.rm = TRUE)
    new_row <- data.frame(gene = gene, Value = var_value)
    result_gene_morans <- rbind(result_gene_morans, new_row)
  }

  out_subdir <- file.path(output_dir, index_slice)
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)
  out_file <- file.path(out_subdir, paste0("gene_morans_", n_quadrants, ".txt"))
  write.table(result_gene_morans, out_file, row.names = FALSE, quote = FALSE, sep = "\t", col.names = TRUE)
  cat("Wrote", out_file, "\n")
}
