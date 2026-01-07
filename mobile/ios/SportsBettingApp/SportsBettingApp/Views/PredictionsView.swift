import SwiftUI

struct PredictionsView: View {
    @StateObject var viewModel: PredictionsViewModel
    @Environment(\.openURL) private var openURL

    var body: some View {
        List {
            if viewModel.isLoading {
                HStack {
                    Spacer()
                    ProgressView()
                    Spacer()
                }
            }

            ForEach(viewModel.predictions) { prediction in
                VStack(alignment: .leading, spacing: 8) {
                    Text("\(prediction.awayTeam ?? "Away") @ \(prediction.homeTeam ?? "Home")")
                        .font(.headline)
                    if let recommendation = prediction.primaryRecommendation, !recommendation.isEmpty {
                        Text("Recommendation: \(recommendation)")
                    }
                    if let confidence = prediction.confidence, !confidence.isEmpty {
                        Text("Confidence: \(confidence)")
                    }
                    if let confidenceTier = prediction.confidenceTier, !confidenceTier.isEmpty {
                        Text("Confidence Tier: \(confidenceTier)")
                    }
                    if let valueTier = prediction.valueTier, !valueTier.isEmpty {
                        Text("Value Tier: \(valueTier)")
                    }
                    if let price = prediction.price, !price.isEmpty {
                        Text("Price: \(price)")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.vertical, 4)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Predictions")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Open CSV") {
                    openURL(viewModel.downloadURL)
                }
            }
        }
        .task {
            await viewModel.fetchPredictions()
        }
        .alert("Something went wrong", isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { _ in viewModel.errorMessage = nil }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(viewModel.errorMessage ?? "Please try again.")
        }
    }
}

#Preview {
    NavigationStack {
        PredictionsView(viewModel: PredictionsViewModel(runId: "1"))
    }
}
