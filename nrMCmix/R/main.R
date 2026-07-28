#FUNCTION TO GET THE CONFIGURATION VECTORS FROM THE FILENAME AND DELETE IT
#' Read and return label allocation draws from a sampler chain
#'
#' Loads the allocation vectors stored in a chain-specific CSV file produced
#' by a sampler, removes the file after reading, and returns the contents as
#' a matrix.
#'
#' @param sampler A sampler object containing a `filename` field specifying
#'   the directory where chain output files are stored.
#' @param chain_id Integer identifying the chain to read. Defaults to `1`.
#'
#' @return A matrix containing the allocation vectors (label draws) read from
#'   the corresponding `confs<chain_id>.csv` file.
#'
#' @details
#' The function expects a file named
#' `confs<chain_id>.csv` to exist inside the directory specified by
#' `sampler$filename`. After successfully reading the file, the file is deleted
#' using `file.remove()`.
#'
#' @examples
#' \dontrun{
#' sampler <- list(filename = "output")
#'
#' labels <- get_labels_draws(sampler, chain_id = 1)
#' }
#'
#' @export
get_labels_draws <- function(sampler, chain_id = 1){
  
  #get the allocation vectors
  filename <- paste0(sampler$filename,"/confs",chain_id,".csv")
  confs <- as.matrix(read.csv(filename, header = FALSE))
  colnames(confs) <- rownames(confs) <- NULL
  
  #delete the file
  file.remove(filename)
  
  #return the labels
  confs
  
}
