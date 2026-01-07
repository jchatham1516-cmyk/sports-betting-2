import SwiftUI

struct RunDetailView: View {
    @StateObject var viewModel: RunDetailViewModel
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
                    HStack {
                        Text(prediction.primaryRecommendation ?? "No recommendation")
                        Spacer()
                        Text(prediction.confidenceTier ?? prediction.valueTier ?? "")
                            .foregroundColor(.secondary)
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
        .navigationTitle("Run Details")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Download CSV") {
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
        RunDetailView(viewModel: RunDetailViewModel(runId: 1))
    }
}
