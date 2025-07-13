# Java Neural Network 🧠

Implementação simples de rede neural em Java, inspirada em tutoriais como os da série *The Coding Train* (Daniel Shiffman).

## Conteúdo

* 💻 Implementação pura em Java
* 🔁 Feedforward + Backpropagation
* ⚙️ Configuração flexível: entradas, camadas ocultas, saídas
* 🎯 Funções de ativação: Sigmoid (padrão), Tanh, ReLU
* 📉 Taxa de aprendizado ajustável
* 💾 Treino, mutação e crossover (para uso com algoritmos genéticos)
* 📂 Serialização/deserialização via JSON

---

## Instalação

Requisitos:

* Java 8+
* Maven

Para compilar o projeto:

```bash
git clone https://github.com/Sobrukai404/java-neural-network.git
cd java-neural-network
mvn install
mvn package
```

Isso irá gerar os `.jar` em `target/`.

---

## Exemplo de uso

```java
// Rede 2 entradas → 4 neurônios ocultos → 1 saída
NeuralNetwork nn = new NeuralNetwork(2, 4, 1);

// Ajuste da taxa de aprendizado
nn.setLearningRate(0.05);

// Treinamento (ex: XOR)
double[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
double[][] targets = {{0},{1},{1},{0}};
nn.train(inputs, targets, epochs:10000);

// Predição
double[] resultado = nn.guess(new double[]{1,0});
System.out.println(resultado[0]);
```

---

## Salvando e carregando redes

```java
// Salvar em JSON
nn.writeToFile("minhaRede");

// Carregar rede de arquivo
NeuralNetwork nn2 = NeuralNetwork.readFromFile("minhaRede.json");
```

---

## Algoritmos genéticos

```java
NeuralNetwork copiar = nn.copy();
NeuralNetwork mergeia = nn.merge(outro, 0.5);
nn.mutate(0.1);
```

---

## Funções de ativação

* Sigmoid (padrão)
* Tanh
* ReLU

Use `nn.setActivationFunction(...)` para alterar.

---

## Roadmap / TODO

* Implementar softmax
* Mais opções para crossover/mutação
* Testes unitários (JUnit)
* Documentação automática (Javadoc)
* Normalização de pesos/biases
* Adicionar exemplos em Processing / interfaces gráficas

---

## Inspirações & referências

* *Basic Neural Network Library* (Kim‑Marcel), inspirada pela série de Daniel Shiffman (The Coding Train) ([github.com][1], [stackoverflow.com][2], [medium.com][3])
* Conceitos de backpropagation e funções de ativação explicados pelo artigo "Building a Neural Network from Scratch in Java" ([medium.com][3])

---

## Licença

MIT

[1]: https://github.com/kim-marcel/basic_neural_network?utm_source=chatgpt.com "A very basic Java Neural Network Library. - GitHub"
[2]: https://stackoverflow.com/questions/5750061/javanns-parsing-the-created-neural-network?utm_source=chatgpt.com "java - JavaNNS - Parsing the created Neural Network - Stack Overflow"
[3]: https://medium.com/%40nirmal1067/building-a-neural-network-from-scratch-in-java-a-step-by-step-guide-4d06afc5ad1d?utm_source=chatgpt.com "Building a Neural Network from Scratch in Java - Medium"
