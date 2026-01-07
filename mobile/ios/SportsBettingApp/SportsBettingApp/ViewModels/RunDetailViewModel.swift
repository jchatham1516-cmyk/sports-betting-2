import Foundation

@MainActor
final class RunDetailViewModel: ObservableObject {
    @Published var predictions: [PredictionRow] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    let runId: Int

    init(runId: Int) {
        self.runId = runId
    }

    func fetchPredictions() async {
        isLoading = true
        errorMessage = nil
        do {
            let response: [PredictionRow] = try await APIClient.get("/api/runs/\(runId)/predictions")
            predictions = response
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    var downloadURL: URL {
        AppConfig.baseURL.appendingPathComponent("/api/runs/\(runId)/download/predictions.csv")
    }
}
