import React, { useState } from 'react';

const TransactionForm = () => {
    const [transactions, setTransactions] = useState([
        { CustomerId: '', TransactionId: '', TransactionDate: '', Amount: '' }
    ]);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleChange = (index, e) => {
        const { name, value } = e.target;
        const updatedTransactions = [...transactions];
        updatedTransactions[index][name] = value;
        setTransactions(updatedTransactions);
    };

    const addTransaction = () => {
        setTransactions([...transactions, { CustomerId: '', TransactionId: '', TransactionDate: '', Amount: '' }]);
    };

    const removeTransaction = (index) => {
        const updatedTransactions = transactions.filter((_, i) => i !== index);
        setTransactions(updatedTransactions);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(transactions)
            });

            if (!response.ok) {
                throw new Error('Failed to fetch prediction');
            }

            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error fetching prediction:', error);
            setResult({ error: 'Failed to fetch prediction' });
        }
        setLoading(false);
    };

    return (
        <div className="max-w-4xl mx-auto p-6 bg-white shadow-md rounded-lg mt-10">
            <h1 className="text-3xl font-bold text-center mb-6">Credit Scoring Prediction</h1>
            <form onSubmit={handleSubmit} className="space-y-4">
                {transactions.map((transaction, index) => (
                    <div key={index} className="border p-4 rounded-lg mb-4">
                        <h2 className="text-lg font-semibold mb-2">Transaction {index + 1}</h2>
                        <div className="mb-3">
                            <label className="block text-sm font-medium mb-1">Customer ID</label>
                            <input
                                type="number"
                                name="CustomerId"
                                className="w-full border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                value={transaction.CustomerId}
                                onChange={(e) => handleChange(index, e)}
                                required
                            />
                        </div>
                        <div className="mb-3">
                            <label className="block text-sm font-medium mb-1">Transaction ID</label>
                            <input
                                type="number"
                                name="TransactionId"
                                className="w-full border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                value={transaction.TransactionId}
                                onChange={(e) => handleChange(index, e)}
                                required
                            />
                        </div>
                        <div className="mb-3">
                            <label className="block text-sm font-medium mb-1">Transaction Date</label>
                            <input
                                type="date"
                                name="TransactionDate"
                                className="w-full border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                value={transaction.TransactionDate}
                                onChange={(e) => handleChange(index, e)}
                                required
                            />
                        </div>
                        <div className="mb-3">
                            <label className="block text-sm font-medium mb-1">Amount</label>
                            <input
                                type="number"
                                name="Amount"
                                className="w-full border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                                value={transaction.Amount}
                                onChange={(e) => handleChange(index, e)}
                                required
                            />
                        </div>
                        {index > 0 && (
                            <button
                                type="button"
                                className="text-red-500 hover:underline"
                                onClick={() => removeTransaction(index)}
                            >
                                Remove Transaction
                            </button>
                        )}
                    </div>
                ))}
                <button
                    type="button"
                    className="w-full bg-green-500 text-white py-2 rounded-lg hover:bg-green-600 transition duration-200"
                    onClick={addTransaction}
                >
                    Add Another Transaction
                </button>
                <button
                    type="submit"
                    className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 transition duration-200"
                >
                    {loading ? 'Submitting...' : 'Submit'}
                </button>
            </form>

            {result && (
                <div className="mt-6 p-4 bg-gray-100 rounded-lg">
                    {result.error ? (
                        <p className="text-red-500">{result.error}</p>
                    ) : (
                        result.map((res, index) => (
                            <div key={index} className="mb-4">
                                <h3 className="text-lg font-semibold">Customer ID: {res.CustomerId}</h3>
                                <p><strong>Recency:</strong> {res.calculated_rfms.Recency} days</p>
                                <p><strong>Frequency:</strong> {res.calculated_rfms.Frequency} transactions</p>
                                <p><strong>Monetary:</strong> ${res.calculated_rfms.Monetary}</p>
                                <p><strong>Size:</strong> ${res.calculated_rfms.Size}</p>
                                <p><strong>Prediction:</strong> {res.prediction === 1 ? 'Good' : 'Bad'}</p>
                                <p><strong>Probability:</strong> {(res.probability * 100).toFixed(2)}%</p>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
};

export default TransactionForm;
