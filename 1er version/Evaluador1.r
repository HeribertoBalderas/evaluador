# Inicio del proyecto
library(dplyr)

# Base de datos
venta = read.csv(file.choose())
head(venta)
venta = venta %>% 
  mutate(precio = Valor / Volumen)
head(venta)

# Filtro de datos
Cloro_2L = filter(venta, Sku == "Cloralex Regular 2lts")
head(Cloro_2L)
attach(Cloro_2L)

# Modelo estadistico
m1 = lm(precio~Volumen)
summary(m1)

# Grafico 
plot(precio, Volumen, xlab = "Precio x pieza", 
     ylab = "Venta en piezas", pch = 16)
abline(lsfit(precio, Volumen))

library(ggplot2)
ggplot(data = Cloro_2L, aes(x = precio, y = Volumen)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color = "firebrick") +
  theme_bw()
