library(ggplot2)


plot_frame = rbind(read.table("data/trace/rsgld-sparse-trace.dat"), 
                   read.table("data/trace/cir-sparse-trace.dat"),
                   read.table("data/trace/exact-sparse-trace.dat"))
plot_frame$Method = rep(c("SGRLD", "SCIR", "Exact"), each = 10^3)
plot_frame = plot_frame[,c(5,11)]
colnames(plot_frame) = c("Omega", "Method")
# Boxplots of SCIR, SGRLD and Exact on Sparse Data
p = ggplot(plot_frame, aes(x = Method, y = Omega, fill = Method)) +
        geom_boxplot() +
        scale_y_log10()
ggsave("plots/cir-trace.pdf", width = 8, height = 3)
# Boxplots of SGRLD and Exact on Sparse Data
plot_frame = subset(plot_frame, Method != "SCIR")
p = ggplot(plot_frame, aes(x = Method, y = Omega, fill = Method)) +
        geom_boxplot() +
        scale_y_log10()
ggsave("plots/rsgld-trace.pdf", width = 7, height = 3)


plot_frame = read.csv('data/sparse_n_processed.csv')
plot_frame$Method = as.character(plot_frame$Method)
plot_frame$Method[plot_frame$Method == 'RSGLD'] = 'SGRLD'
# Convert minibatch size to proportions
plot_frame$Minibatch.Size = plot_frame$Minibatch.Size / 1000
p = ggplot(plot_frame, aes(x = Minibatch.Size, y = KS.Distance, ymin = Lower, 
            ymax = Upper, fill = Method)) +
        geom_ribbon(color = 'black') +
        scale_x_log10() +
        xlab('Minibatch Size (log scale)') +
        ylab('KS Distance')
ggsave("plots/sparse-ks.pdf", width = 8, height = 3)


plot_frame = read.csv('data/dense_n_processed.csv')
plot_frame$Method = as.character(plot_frame$Method)
plot_frame$Method[plot_frame$Method == 'RSGLD'] = 'SGRLD'
# Convert minibatch size to proportions
plot_frame$Minibatch.Size = plot_frame$Minibatch.Size / 1000
p = ggplot(plot_frame, aes(x = Minibatch.Size, y = KS.Distance, ymin = Lower, 
            ymax = Upper, fill = Method)) +
        geom_ribbon(color = 'black', alpha = .5) +
        scale_x_log10() +
        xlab('Minibatch Size (log scale)') +
        ylab('KS Distance')
ggsave("plots/dense-ks.pdf", width = 8, height = 3)
