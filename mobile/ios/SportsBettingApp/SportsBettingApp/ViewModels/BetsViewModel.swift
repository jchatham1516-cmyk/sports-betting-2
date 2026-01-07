import Foundation

@MainActor
final class BetsViewModel: ObservableObject {
    @Published var selectedSport: Sport = .nba
    @Published var selectedDate: Date = Date()
    @Published var bets: [TrackedBetRow] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    func fetchBets() async {
        isLoading = true
        errorMessage = nil
        let dateString = dateFormatter.string(from: selectedDate)
        let queryItems = [
            URLQueryItem(name: "date", value: dateString),
            URLQueryItem(name: "sport", value: selectedSport.rawValue)
        ]
        do {
            let response: [TrackedBetRow] = try await APIClient.get("/api/bets", queryItems: queryItems)
            bets = response
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
