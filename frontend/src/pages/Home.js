import React from 'react';
import { Link } from 'react-router-dom';
import '../css/Home.css';

const Home = () => {
  return (
    <div className="home-container">
      <div className="home-content">
        <div className="home-header">
          <h1>Assistant Code Pénal Burundais</h1>
          <p>Votre assistant intelligent spécialisé dans le droit pénal du Burundi</p>
        </div>
        
        <div className="home-features">
          <div className="feature">
            <div className="feature-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
              </svg>
            </div>
            <h3>Expertise Juridique</h3>
            <p>Accès instantané aux articles du Code pénal burundais avec des explications précises</p>
          </div>
          
          <div className="feature">
            <div className="feature-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
            </div>
            <h3>Réponses Fiables</h3>
            <p>Informations basées uniquement sur le Code pénal officiel du Burundi</p>
          </div>
          
          <div className="feature">
            <div className="feature-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
              </svg>
            </div>
            <h3>Interface Simple</h3>
            <p>Posez vos questions en français et obtenez des réponses claires et précises</p>
          </div>
        </div>
        
        <div className="home-actions">
          <Link to="/chat" className="start-chat-btn">
            Commencer une conversation
          </Link>
        </div>
        
        <div className="home-disclaimer">
          <p>
            <strong>Note importante :</strong> Cet assistant fournit des informations juridiques, 
            pas des conseils juridiques. Consultez un professionnel du droit pour des cas spécifiques.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Home;