import SwiftUI

struct BetsView: View {
    @StateObject private var viewModel = BetsViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Filters") {
                    Picker("Sport", selection: $viewModel.selectedSport) {
                        ForEach(Sport.allCases) { sport in
                            Text(sport.rawValue.uppercased()).tag(sport)
                        }
                    }
                    .pickerStyle(.segmented)

                    DatePicker("Date", selection: $viewModel.selectedDate, displayedComponents: .date)
                }

                Section {
                    Button {
                        Task {
                            await viewModel.fetchBets()
                        }
                    } label: {
                        HStack {
                            Spacer()
                            if viewModel.isLoading {
                                ProgressView()
                            } else {
                                Text("Load Bets")
                                    .fontWeight(.semibold)
                            }
                            Spacer()
                        }
                    }
                    .disabled(viewModel.isLoading)
                }

                Section("Tracked Bets") {
                    if viewModel.bets.isEmpty {
                        Text("No bets found for the selected filters.")
                            .foregroundColor(.secondary)
                    } else {
                        ForEach(viewModel.bets) { bet in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(bet.matchup ?? "Matchup")
                                    .font(.headline)
                                Text("Pick: \(bet.pick ?? "-") • Market: \(bet.market ?? "-")")
                                    .font(.subheadline)
                                Text("Units: \(bet.units?.description ?? "-") • Result: \(bet.result ?? "-")")
                                    .font(.footnote)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
            .navigationTitle("Tracked Bets")
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
}

#Preview {
    BetsView()
}
