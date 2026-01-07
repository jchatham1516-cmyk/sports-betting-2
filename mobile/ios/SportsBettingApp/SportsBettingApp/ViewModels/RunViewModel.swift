import Foundation

@MainActor
final class RunViewModel: ObservableObject {
    @Published var selectedSport: Sport = .nba
    @Published var selectedDate: Date = Date()
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var runResponse: RunCreateResponse?

    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    func createRun() async {
        isLoading = true
        errorMessage = nil
        do {
            let request = RunCreateRequest(
                sport: selectedSport.rawValue,
                gameDate: dateFormatter.string(from: selectedDate),
                settings: [:]
            )
            let response: RunCreateResponse = try await APIClient.post("/api/runs", body: request)
            runResponse = response
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
